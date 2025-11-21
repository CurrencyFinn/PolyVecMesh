from xml.etree import ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

### TODO
### Functionality
# Multiregion + VTK coloring
# Smarter selection of cells that are outside the plane when using implicit clip, \
#       e.g. before making cells selecting points that have issues to filter off, \
#       perhaps create a distance inveral for maxDistanceOffSlice or as a coordinate
# Auto select maxDistanceOffSlice -> Own slicing tool pretty much
# Only select frontal faces -> small details motorBike case issues projection, where boundary layers collapse

### Optimalization
# Skip faces/edges with normal far away from the posed normal
# Switch hexehedrals to facewise assigning, can then use uniqueness to filter that off with faces from polyhedrals


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
    direction : int, default=1
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
    edges : tuple
        Hardcoded hexahedral edges.
    """
    def __init__(
            self, fileName, sliceOrigin=None, 
            debug=0, normalPlaneDirection=None, tol=1e-3, direction=1):
        
        self.fileName = fileName
        self.normalPlaneDirection = normalPlaneDirection
        self.tol = tol
        self.debug = debug
        self.sliceOrigin = sliceOrigin
        self.planeSide= direction
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

        self.edges = (
            [0,1],[1,2],[2,3],[3,0],  # bottom face
            [4,5],[5,6],[6,7],[7,4],  # top face
            [0,4],[1,5],[2,6],[3,7]   # vertical edges
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

            for _ in range(nFaces): # Go through each face closing
                nVerts = self.faces[idx]
                verts = self.faces[idx+1:idx+1+nVerts]
                pts2D = np.delete(self.points[verts], self.normalPlaneDirection, axis=1)
                pts2D = np.vstack([pts2D, pts2D[0]])  # close this face polygon
                self.allCellCollection.append(pts2D)
                if self.debug:
                    self.colors.append('red')
                idx += 1 + nVerts

        # VTK_HEXAHEDRON (12)
        if self.cellTypes[cellIndex] == 12 and len(cellPoints) == 8:
            for e in self.edges:
                start3D = cellPoints[e[0]]
                end3D   = cellPoints[e[1]]

                # (optional) Keep only edges on one side of the slicing plane: uncomment if needed ### led to 30% decrease in cells
                if self.planeSide == 1:
                    if start3D[self.normalPlaneDirection] < self.sliceCoordinate:
                        continue
                    if end3D[self.normalPlaneDirection] < self.sliceCoordinate:
                        continue
                elif self.planeSide == -1:
                    if start3D[self.normalPlaneDirection] > self.sliceCoordinate:
                        continue
                    if end3D[self.normalPlaneDirection] > self.sliceCoordinate:
                        continue

                # Skip edges that run along the normalPlaneDirection (i.e. vertical edges relative to slicing plane), decrease number of plot output # Leads to very low amount of decrease
                delta = np.abs(end3D[self.normalPlaneDirection] - start3D[self.normalPlaneDirection])
                if delta > self.tol:
                    continue

                # Project both endpoints to 2D (drop normalPlaneDirection)
                start2D = np.delete(start3D, self.normalPlaneDirection)
                end2D   = np.delete(end3D,   self.normalPlaneDirection)

                # Store segment
                seg = np.vstack([start2D, end2D])
                self.allCellCollection.append(seg)
                if self.debug:
                    self.colors.append('black')
        else:
            Warning("Unsupported VTK type skipping")

    @staticmethod
    def unique_ragged(list_):
        """
        Helper function to find unique arrays in a list without sorting.

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

        # reduce the number of draw calls by only selecting unique points some may overlap in normalPlaneDirection ### led to 50% decrease in draw calls
        self.allCellCollection = self.unique_ragged(self.allCellCollection) 
        
        return self.allCellCollection

    def debug_plot(self):
        """Example function to visualize and debug the extracted VTU mesh slice."""
        self.debug = 1 # overwrite the debug option
        _, ax = plt.subplots(figsize=(10,10))
        meshLines = self.createCollection()
        cellLineCollection = LineCollection(meshLines, colors=self.colors, linewidths=0.5)
        ax.add_collection(cellLineCollection, autolim=True)
        ax.autoscale(enable=True, tight=True)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

