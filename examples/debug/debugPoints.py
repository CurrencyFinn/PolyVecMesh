import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Your points
pts = np.array([[0.00020441028755158186, -0.027565188705921173, 7.237240060931072e-05,],
 [0.00020213727839291096, -0.027584988623857498, 9.909187792800367e-05,],
 [0.00024894249509088695, -0.02749639004468918, 0.00019203239935450256,],
 [0.0002510943741071969, -0.027475930750370026, 0.0001658046239754185,]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
ax.scatter(x, y, z, s=50)

# Labels
for i, (xi, yi, zi) in enumerate(pts):
    ax.text(xi, yi, zi, f"{i}", fontsize=12)

# Draw lines connecting the points (closed loop)
loop = np.vstack([pts, pts[0]])     # append first point to close shape
ax.plot(loop[:,0], loop[:,1], loop[:,2])

# Axes labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()