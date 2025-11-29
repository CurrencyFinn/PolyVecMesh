from xml.etree import ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import math
from numba import jit

### TODO
### Functionality
# Multiregion + VTK coloring
# Smarter selection of cells that are outside the plane when using implicit clip, \
#       e.g. before making cells selecting points that have issues to filter off, \
#       perhaps create a distance inveral for maxDistanceOffSlice or as a coordinate
# Auto select maxDistanceOffSlice -> Own slicing tool pretty much
### Bug
# Debug plot coloring not correct as removal of some of the faces, arrays are inconisistent

class PolyVecMesh:
    """
    Load a VTU mesh file and extract 2D slices for visualization in matplotlib.

    Parameters
    ----------
    fileName : str
        Path to the VTU file.
    sliceOrigin : tuple or None, default=None
        Origin of the slicing plane (x, y, z). Only the component along
        `normalPlaneDirection` is used. If None, computed from mesh midpoint.
    debug : int, default=0
        Enable debug mode for coloring and labeling of cells.
    normalPlaneDirection : int or None, default=None
        Axis normal to the slicing plane: 0=x, 1=y, 2=z. Auto-selected if None.
    tol : float, default=1e-3
        Tolerance for filtering hexahedral edges along the normal direction.
    planeSide : int, default=1
        Side of the plane to include: 1 = along normal, -1 = opposite.

    Attributes
    ----------
    allCellCollection : list
        Collection of 2D points per face (polyhedra) or per cell (hexahedra)
        to be drawn with matplotlib LineCollection.
    colors : list
        List of colors corresponding to `allCellCollection` segments.
    nCells : int
        Number of cells in the mesh.
    points : ndarray
        Array of mesh points (coordinates).
    connectivity : ndarray
        Cell connectivity array (point indices per cell).
    offsets : ndarray
        Offsets into the connectivity array per cell.
    cellTypes : ndarray
        Array of VTK cell types.
    sliceCoordinate : float
        Coordinate along `normalPlaneDirection` defining the slicing plane.
    faces : ndarray
        Array of face definitions for polyhedral cells.
    faceOffsets : ndarray
        Offsets into the `faces` array per cell.
    facesHexaHedral : np.ndarray (6,4)
        Array of hexahedral faces. Each row corresponds to a face of the hexahedron, 
        and contains the indices of the four vertices that form that face.
    """
    def __init__(
            self, fileName, sliceOrigin=None, 
            debug=0, normalPlaneDirection=None, tol=1e-3, planeSide=1):
        
        self.fileName = fileName
        self.normalPlaneDirection = normalPlaneDirection
        self.tol = tol
        self.epsilon = 0.1
        self.debug = debug
        self.sliceOrigin = sliceOrigin
        self.planeSide= planeSide
        self.allCellCollection = []
        
        self.nCells = None
        self.points = None
        self.connectivity = None
        self.offsets = None
        self.cellTypes = None
        self.sliceCoordinate = None
        self.faces = None
        self.faceOffsets = None
        self.colors = None

        self.facesHexaHedral = np.array(
            ((0,1,2,3),   # Face 0  bottom
            (4,5,6,7),    # Face 1  top
            (0,4,5,1),    # Face 2  side
            (1,5,6,2),    # Face 3  side
            (2,6,7,3),    # Face 4  side
            (3,7,4,0)),   # Face 5  side
            dtype=np.int8
        )

        self.loadVTK()

    def loadVTK(self):   
        """Load VTU file and parse points, connectivity, offsets, cell types, faces, and face offsets from XML format."""
        # Load VTU XML
        tree = ET.parse(self.fileName)
        root = tree.getroot()
        meshData = root[0].find("Piece")

        self.nCells = int(meshData.attrib['NumberOfCells'])

        # Read points
        pointDataXML = meshData.find(".//DataArray[@Name='Points']")
        nComponents = int(pointDataXML.attrib["NumberOfComponents"])
        pointTypeStr = pointDataXML.attrib["type"]

        pointTyping = np.float32 if pointTypeStr == "Float32" else np.float64
        txt = pointDataXML.text.strip().split() 
        pointData = np.array(txt, dtype=pointTyping)   
        self.points = pointData.reshape(-1, nComponents)

        # Auto compute the normal direction of plane taken and origin
        if self.normalPlaneDirection == None:
            ranges = self.points.max(axis=0) - self.points.min(axis=0)
            self.normalPlaneDirection = np.argmin(ranges)
        if self.sliceOrigin == None:
            mins = self.points.min(axis=0)
            maxs = self.points.max(axis=0)
            midpoint = 0.5 * (mins + maxs)
            self.sliceOrigin = midpoint

        # Read connectivity
        connectivityDataXML = meshData.find(".//DataArray[@Name='connectivity']")
        connecitivityTyping = np.int64 if connectivityDataXML.attrib["type"] == "Int64" else np.int32
        self.connectivity = np.fromstring(connectivityDataXML.text, sep=' ', dtype=connecitivityTyping)

        # Read offsets
        offsetsDataXML = meshData.find(".//DataArray[@Name='offsets']")
        offsetsTyping = np.int64 if offsetsDataXML.attrib["type"] == "Int64" else np.int32
        self.offsets = np.fromstring(offsetsDataXML.text, sep=' ', dtype=offsetsTyping)
        if self.offsets.size == 0 or self.offsets[0] != 0: # Safe indexing 
            self.offsets = np.insert(self.offsets, 0, 0)

        # Read cell types
        cellTypesXML = meshData.find(".//DataArray[@Name='types']")
        self.cellTypes = np.fromstring(cellTypesXML.text, sep=' ', dtype=np.uint8)

        # Read faces
        facesDataXML = meshData.find(".//DataArray[@Name='faces']")
        facesType = np.int64 if facesDataXML.attrib["type"] == "Int64" else np.int32
        self.faces = np.fromstring(facesDataXML.text, sep=' ', dtype=facesType)

        # Read face offsets
        faceOffsetsXML = meshData.find(".//DataArray[@Name='faceoffsets']")
        offsetsType = np.int64 if faceOffsetsXML.attrib["type"] == "Int64" else np.int32
        self.faceOffsets = np.fromstring(faceOffsetsXML.text, sep=' ', dtype=offsetsType)


    def generateCellLines(self, cellPoints, cellIndex):
        """Extracts and projects the 3D points of a given cell (polyhedron or hexahedron) onto 2D, generating line segments for visualization and optionally assigning debug colors"""
        # VTK_POLYHEDRON (42) ### Do not use cellPoints but create all faces
        if self.cellTypes[cellIndex] == 42:
            faceOffsetValue = self.faceOffsets[cellIndex]
            if faceOffsetValue >= len(self.faces):
                return
            nFaces = self.faces[faceOffsetValue]
            idx = faceOffsetValue + 1   # first face starts after nFaces

            allPts3D = []
            for _ in range(nFaces): # Go through each face closing
                nVerts = self.faces[idx]
                verts = self.faces[idx+1:idx+1+nVerts]
                pts3D = self.points[verts]
                allPts3D.append(pts3D)
                normal = self.plane_normal(pts3D)  
                if self.is_axis_aligned(normal[self.normalPlaneDirection], strict=False): # TODO leaves unwanted slivers motorBike case
                    pts3D = np.vstack([pts3D, pts3D[0]])  # close this face polygon
                    self.allCellCollection.append(pts3D)
                    if self.debug:
                        self.colors.append('red')
                idx += 1 + nVerts

            '''
            # The normal is always correctly computed so no need to adjust this in polyhedron

            allFaceValues = np.concatenate(allPts3D).reshape(-1, 3)
            cell_center = allFaceValues.mean(axis=0)
            for pts3D in allPts3D:
                normal = self.plane_normal(pts3D)
                face_center = pts3D.mean(axis=0)
                if np.dot(normal, face_center - cell_center) < 0:
                    pts3D = pts3D[::-1]  # flip winding
                    normal = -normal   
                    # print("wrong normal")
            '''

        # VTK_HEXAHEDRON (12)
        if self.cellTypes[cellIndex] == 12 and len(cellPoints) == 8:
            cell_center = cellPoints.mean(axis=0)
            for face in self.facesHexaHedral: # Go through each face closing
                pts3D = cellPoints[face]
                normal = self.plane_normal(pts3D)
                face_center = pts3D.mean(axis=0)
                if np.dot(normal, face_center - cell_center) < 0:
                    pts3D = pts3D[::-1]  # flip winding
                    normal = -normal   
                    #print("wrong normal")

                if self.is_axis_aligned(normal[self.normalPlaneDirection], strict=True, planeSide=self.planeSide): 
                    pts3D = np.vstack([pts3D, pts3D[0]])  # close the hexahedral
                    self.allCellCollection.append(pts3D)
                    if self.debug:
                        self.colors.append('black')
        else:
            Warning("Unsupported VTK type skipping")

    @staticmethod
    def unique_ragged(list_):
        """
        Helper function to find unique arrays in a list without sorting. 
        Keeps first occurence deletes duplicates after first occurence.

        Parameters
        ----------
        list_ : list of numpy.ndarray
            List of arrays (possibly ragged) to filter for uniqueness.

        Returns
        -------
        unique_list : list of numpy.ndarray
            List containing only unique arrays from the input list, preserving order.
        """
        seen = set()
        unique_list = []
        for arr in list_:
            h = hash(arr.tobytes())
            if h not in seen:
                seen.add(h)
                unique_list.append(arr)
        return unique_list
    
    @staticmethod
    @jit(nopython=True) 
    def plane_normal(facePoints):
        """
        Compute a normal for a polygon/hexahedral robustly.

        Parameters
        ----------
        facePoints : numpy.ndarray (N,3) of 3D coordinates
            Array of 3D points on given face.

        Returns
        -------
        normal : numpy.ndarray (3)
            Normal unit vector, or [0,0,0] if all points are collinear.
        """

        nPoints = facePoints.shape[0]
        
        for i in range(nPoints - 2):
            A = facePoints[i]
            B = facePoints[i + 1]
            C = facePoints[i + 2]

            v1 = B - A
            v2 = C - A

            # cross product
            nx = v1[1]*v2[2] - v1[2]*v2[1]
            ny = v1[2]*v2[0] - v1[0]*v2[2]
            nz = v1[0]*v2[1] - v1[1]*v2[0]

            norm = math.sqrt(nx*nx + ny*ny + nz*nz)

            if norm > 1e-8: 
                inv = 1.0 / norm
                normal = np.array([nx*inv, ny*inv, nz*inv], dtype=np.float32)
                return normal
            
        print("Warning: zero normal face detected")
        # all triples were degenerate
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return normal

    @staticmethod
    @jit(nopython=True)
    def is_axis_aligned(normalComponent, tol=0.25, strict=False, planeSide=0.0):
        """
        Check if a component of a normal vector is aligned with an axis.

        Parameters
        ----------
        normalComponent : float
            The component of the normal vector along a chosen coordinate axis.
        tol : float, default=0.25
            Tolerance for alignment
        strict : bool, optional, default=False
            If True, checks alignment relative to `planeSide`.  
            If False, checks if the face lies on the plane along the axis 
            irrespective of the normal's direction (front or back).  
        planeSide : float, optional, default=0.0
            Reference value for strict alignment, can be -1 or 1. Only used if `strict=True`.

        Returns
        -------
        aligned : bool
            True if the normal component is considered aligned with the axis 
            within the specified tolerance, False otherwise.
        """

        if strict:
            return abs(normalComponent - planeSide) <= tol
        else:
            return abs(normalComponent - 1.0) <= tol or abs(normalComponent + 1.0) <= tol


    def createCollection(self, maxDistanceOffSlice=np.inf):
        """
        Extracts 2D line segments for cells near the slicing plane.

        Parameters
        ----------
        maxDistanceOffSlice : float, optional
            Maximum allowed distance from the slicing plane for including a cell.
            Cells farther than this are skipped. Smaller cells may be disproportionately
            excluded when this value is used.

        Returns
        -------
        list of np.ndarray
            List of 2D line segments representing the cell edges for plotting.
        """

        if self.debug:
            self.colors = []

        self.maxDistanceOffSlice = maxDistanceOffSlice
        # Build line segments for 2D top view
        self.sliceCoordinate =  self.sliceOrigin[self.normalPlaneDirection]

        for i in range(self.nCells):
            # Get cell-local connectivity (list of global point ids for this cell) 
            cell_start = self.offsets[i]
            cell_end = self.offsets[i+1]
            cellConnectivity = self.connectivity[cell_start:cell_end]
            cellPoints = self.points[cellConnectivity]

            cellSliceCoordinate = cellPoints[:, self.normalPlaneDirection]
            notInSlice = (cellSliceCoordinate > self.sliceCoordinate + self.maxDistanceOffSlice) | \
                    (cellSliceCoordinate < self.sliceCoordinate - self.maxDistanceOffSlice)

            if np.any(notInSlice):
                continue 

            self.generateCellLines(cellPoints, i)
        
        # Sort faces along normalPlaneDirection so that the near-to-far ordering is preserved.
        # unique_ragged() will then eliminate duplicate faces on the opposite side.
        # After deduplication, reverse the list so that rendering draws the nearest faces last.
        averageFaceHeight = np.array([np.average(face[:, self.normalPlaneDirection], axis=0) for face in self.allCellCollection])
        heightMapIndices = np.argsort(averageFaceHeight)
        if self.planeSide == 1:
            heightMapIndices = heightMapIndices[::-1]

        self.allCellCollection = [self.allCellCollection[i] for i in heightMapIndices]
        self.allCellCollection =  [np.delete(face, self.normalPlaneDirection, axis=1) for face in self.allCellCollection]
        
        # reduce the number of draw calls by only selecting unique points some may overlap in normalPlaneDirection
        self.allCellCollection = self.unique_ragged(self.allCellCollection) 
        self.allCellCollection = self.allCellCollection[::-1]


        return self.allCellCollection

    def debug_plot(self):
        """Example function to visualize and debug the extracted VTU mesh slice."""
        self.debug = 1 # overwrite the debug option
        _, ax = plt.subplots(figsize=(10,10))
        meshLines = self.createCollection()
        poly = PolyCollection(meshLines, closed=False, edgecolors=self.colors, facecolors='white', linewidths=0.5)
        ax.add_collection(poly, autolim=True)
        ax.autoscale(enable=True, tight=True)
        ax.set_aspect('equal', adjustable='box')
        plt.show()