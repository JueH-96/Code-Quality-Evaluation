import math

class Solution:
  def canReachCorner(self, xCorner: int, yCorner: int, circles: list[list[int]]) -> bool:
    
    active_circles_data = []
    for i in range(len(circles)):
      x, y, r = circles[i]
      
      # Filter out circles that do not intersect the rectangle [0, xCorner] x [0, yCorner]
      # The point in the rectangle closest to the circle's center (x,y) is (px,py).
      # Clamp x_c to [0, xCorner]
      px = x
      if px < 0: px = 0
      if px > xCorner: px = xCorner
      
      # Clamp y_c to [0, yCorner]
      py = y
      if py < 0: py = 0
      if py > yCorner: py = yCorner
      
      # Squared distance from circle center (x,y) to closest point in rectangle (px,py)
      dist_sq_to_rect_point = (x - px)**2 + (y - py)**2
      
      if dist_sq_to_rect_point <= r**2:
        # Circle intersects or is contained within the rectangle. Add it.
        active_circles_data.append([x, y, r])
    
    N = len(active_circles_data) # Number of active circles
    
    # DSU structure:
    # Parent array for N circles and 4 wall nodes
    parent = list(range(N + 4)) 
    # Sizes for union-by-size heuristic
    sz = [1] * (N + 4) 

    # Node indices for walls:
    L_NODE = N      # Left Wall (x=0)
    B_NODE = N + 1  # Bottom Wall (y=0)
    R_NODE = N + 2  # Right Wall (x=xCorner)
    T_NODE = N + 3  # Top Wall (y=yCorner)

    def find(k: int) -> int:
      if parent[k] == k:
        return k
      parent[k] = find(parent[k])
      return parent[k]

    def union(k1: int, k2: int):
      root_k1 = find(k1)
      root_k2 = find(k2)
      if root_k1 != root_k2:
        # Union by size: attach smaller tree under root of larger tree
        if sz[root_k1] < sz[root_k2]:
          root_k1, root_k2 = root_k2, root_k1 # Ensure root_k1 is representative of the larger set
        parent[root_k2] = root_k1
        sz[root_k1] += sz[root_k2]

    # Connect circles to walls they touch/cross
    for i in range(N):
      x, y, r = active_circles_data[i]
      
      # Connect to Left Wall if circle touches/crosses x=0.
      # Since x, r are positive integers, x - r <= 0 means circle's leftmost point is at or left of x=0.
      if x - r <= 0:
        union(i, L_NODE)
      
      # Connect to Bottom Wall if circle touches/crosses y=0.
      if y - r <= 0:
        union(i, B_NODE)
        
      # Connect to Right Wall if circle touches/crosses x=xCorner.
      # Circle's rightmost point x+r is at or right of x=xCorner.
      if x + r >= xCorner:
        union(i, R_NODE)
        
      # Connect to Top Wall if circle touches/crosses y=yCorner.
      if y + r >= yCorner:
        union(i, T_NODE)

    # Connect intersecting/touching circles
    for i in range(N):
      for j in range(i + 1, N):
        x1, y1, r1 = active_circles_data[i]
        x2, y2, r2 = active_circles_data[j]
        
        # Check if circle i and circle j intersect/touch using squared distances
        # (x1-x2)^2 + (y1-y2)^2 <= (r1+r2)^2
        dist_sq_centers = (x1 - x2)**2 + (y1 - y2)**2
        # (r1+r2) can be up to 2*10^9. (r1+r2)^2 up to 4*10^18.
        # Python handles large integers automatically.
        sum_radii = r1 + r2
        
        if dist_sq_centers <= sum_radii**2:
          union(i, j)
          
    # Check for blocking conditions. If any is true, path is blocked.
    # 1. Left Wall connected to Bottom Wall (origin (0,0) corner blocked)
    if find(L_NODE) == find(B_NODE):
      return False
      
    # 2. Right Wall connected to Top Wall (destination (xCorner,yCorner) corner blocked)
    if find(R_NODE) == find(T_NODE):
      return False
      
    # 3. Left Wall connected to Right Wall (a barrier from left to right)
    if find(L_NODE) == find(R_NODE):
      return False
      
    # 4. Bottom Wall connected to Top Wall (a barrier from bottom to top)
    if find(B_NODE) == find(T_NODE):
      return False
      
    # If none of the blocking conditions are met, a path exists.
    return True