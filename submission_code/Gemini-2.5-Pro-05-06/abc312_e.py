import sys

# Segment Tree constants
EMPTY_VAL = 0    # Represents an empty segment in the tree
MIXED_VAL = -1   # Represents a segment with mixed children values
NO_LAZY_TAG = -2 # Indicates no pending lazy update for a node

class SegmentTree:
    def __init__(self, max_elements): # max_elements is count of leaf cells, e.g., 100 for indices 0..99
        self.max_leaf_idx = max_elements - 1
        self.st_base = 1
        while self.st_base <= self.max_leaf_idx:
            self.st_base *= 2
        
        self.tree = [EMPTY_VAL] * (2 * self.st_base)
        self.lazy = [NO_LAZY_TAG] * (2 * self.st_base)

    def _push(self, node_idx):
        if self.lazy[node_idx] != NO_LAZY_TAG and node_idx < self.st_base: # Not a leaf
            left_child_idx, right_child_idx = 2 * node_idx, 2 * node_idx + 1
            
            self.tree[left_child_idx] = self.lazy[node_idx]
            self.lazy[left_child_idx] = self.lazy[node_idx]
            
            self.tree[right_child_idx] = self.lazy[node_idx]
            self.lazy[right_child_idx] = self.lazy[node_idx]
        
        self.lazy[node_idx] = NO_LAZY_TAG

    def _update_range_recursive(self, node_idx, current_L, current_R, query_L, query_R, val_to_set):
        self._push(node_idx) 

        if current_L > current_R or current_L > query_R or current_R < query_L: 
            return

        if query_L <= current_L and current_R <= query_R: 
            self.tree[node_idx] = val_to_set
            self.lazy[node_idx] = val_to_set
            return

        mid = (current_L + current_R) // 2
        left_child_idx, right_child_idx = 2 * node_idx, 2 * node_idx + 1
        self._update_range_recursive(left_child_idx, current_L, mid, query_L, query_R, val_to_set)
        self._update_range_recursive(right_child_idx, mid + 1, current_R, query_L, query_R, val_to_set)

        # After children are updated, determine this node's value
        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            self.tree[node_idx] = self.tree[left_child_idx]
        else:
            self.tree[node_idx] = MIXED_VAL
            
    def update(self, Z1, Z2, val_to_set): # Updates cells Z1..Z2-1
        if Z1 >= Z2: # Empty range means Z-dimension length is 0.
            return 
        self._update_range_recursive(1, 0, self.st_base - 1, Z1, Z2 - 1, val_to_set)

    def _query_range_recursive(self, node_idx, current_L, current_R, query_L, query_R, result_set):
        self._push(node_idx)

        if current_L > current_R or current_L > query_R or current_R < query_L or self.tree[node_idx] == EMPTY_VAL:
            return

        if self.tree[node_idx] != MIXED_VAL: 
            # Value is self.tree[node_idx]. Since it's not EMPTY_VAL (checked above)
            # and not MIXED_VAL, it must be an ID+1.
            result_set.add(self.tree[node_idx])
            return
        
        mid = (current_L + current_R) // 2
        self._query_range_recursive(2*node_idx, current_L, mid, query_L, query_R, result_set)
        self._query_range_recursive(2*node_idx+1, mid + 1, current_R, query_L, query_R, result_set)

    def query(self, Z1, Z2): # Queries cells Z1..Z2-1
        if Z1 >= Z2: return set() 
        result_set = set()
        self._query_range_recursive(1, 0, self.st_base - 1, Z1, Z2 - 1, result_set)
        return result_set

def solve():
    N = int(sys.stdin.readline())
    raw_cuboids_input = []
    for i in range(N):
        coords = list(map(int, sys.stdin.readline().split()))
        raw_cuboids_input.append({'id': i, 'coords': coords})

    adj = [set() for _ in range(N)]
    
    NUM_CELLS_PER_DIM = 100 

    END_EVENT = 0
    START_EVENT = 1
    
    # Orientation of face relative to the shared plane X0 (or Y0, Z0)
    # If cuboid $i$ has $x_{i2}=X0$, its face at $X0$ is "right-facing". It's on the "left" side of $X0$.
    # If cuboid $j$ has $x_{j1}=X0$, its face at $X0$ is "left-facing". It's on the "right" side of $X0$.
    RIGHT_FACING_ORIENTATION = 0 
    LEFT_FACING_ORIENTATION = 1  

    for dim_idx in range(3): 
        if dim_idx == 0: # Normal X, Sweep Y, Tree Z
            d1_p1, d2_p1, d3_p1 = 0, 1, 2; d1_p2, d2_p2, d3_p2 = 3, 4, 5
        elif dim_idx == 1: # Normal Y, Sweep X, Tree Z
            d1_p1, d2_p1, d3_p1 = 1, 0, 2; d1_p2, d2_p2, d3_p2 = 4, 3, 5
        else: # Normal Z, Sweep X, Tree Y
            d1_p1, d2_p1, d3_p1 = 2, 0, 1; d1_p2, d2_p2, d3_p2 = 5, 3, 4

        # faces_ending_at_coord[X0] stores faces of cuboids $i$ where $x_{i2}=X0$
        # faces_starting_at_coord[X0] stores faces of cuboids $j$ where $x_{j1}=X0$
        faces_ending_at_coord = [[] for _ in range(NUM_CELLS_PER_DIM + 1)] 
        faces_starting_at_coord = [[] for _ in range(NUM_CELLS_PER_DIM + 1)]


        for cb_data in raw_cuboids_input:
            orig_id = cb_data['id']
            c = cb_data['coords']
            
            # current_d1_coord1, current_d2_coord1, current_d3_coord1
            # current_d1_coord2, current_d2_coord2, current_d3_coord2
            # These are min/max coordinates along the permuted axes (d1, d2, d3)
            cd1_1, cd2_1, cd3_1 = c[d1_p1], c[d2_p1], c[d3_p1]
            cd1_2, cd2_2, cd3_2 = c[d1_p2], c[d2_p2], c[d3_p2]
            
            # Store face rectangle (d2,d3 coords) indexed by d1 coordinate
            faces_starting_at_coord[cd1_1].append({'y1':cd2_1, 'y2':cd2_2, 'z1':cd3_1, 'z2':cd3_2, 'id':orig_id})
            faces_ending_at_coord[cd1_2].append({'y1':cd2_1, 'y2':cd2_2, 'z1':cd3_1, 'z2':cd3_2, 'id':orig_id})
            
        # ST for "left group" (cuboids $j$ with $x_{j1}=X0$)
        # ST for "right group" (cuboids $i$ with $x_{i2}=X0$)
        st_left_group = SegmentTree(NUM_CELLS_PER_DIM) 
        st_right_group = SegmentTree(NUM_CELLS_PER_DIM)

        for D1_val in range(1, NUM_CELLS_PER_DIM): # Shared plane coord D1_val can be 1..99
            events = []
            
            for face_data in faces_starting_at_coord[D1_val]: 
                events.append((face_data['y1'], START_EVENT, face_data['z1'], face_data['z2'], face_data['id'], LEFT_FACING_ORIENTATION ))
                events.append((face_data['y2'], END_EVENT,   face_data['z1'], face_data['z2'], face_data['id'], LEFT_FACING_ORIENTATION ))
            
            for face_data in faces_ending_at_coord[D1_val]: 
                events.append((face_data['y1'], START_EVENT, face_data['z1'], face_data['z2'], face_data['id'], RIGHT_FACING_ORIENTATION ))
                events.append((face_data['y2'], END_EVENT,   face_data['z1'], face_data['z2'], face_data['id'], RIGHT_FACING_ORIENTATION ))
            
            if not events: continue
            events.sort() # Sort by y (d2_coord), then by type (END_EVENT=0 before START_EVENT=1)

            for d2_coord, type, d3_coord1, d3_coord2, cuboid_id_orig, face_orientation in events:
                
                if type == START_EVENT:
                    query_st = st_right_group if face_orientation == LEFT_FACING_ORIENTATION else st_left_group
                    update_st = st_left_group if face_orientation == LEFT_FACING_ORIENTATION else st_right_group
                    
                    overlapping_stored_vals = query_st.query(d3_coord1, d3_coord2) 
                    for stored_val in overlapping_stored_vals:
                        other_id_orig = stored_val - 1 # Convert from ID+1 back to original ID
                        adj[cuboid_id_orig].add(other_id_orig)
                        adj[other_id_orig].add(cuboid_id_orig)
                    
                    update_st.update(d3_coord1, d3_coord2, cuboid_id_orig + 1) # Store ID+1
                else: # END_EVENT
                    update_st = st_left_group if face_orientation == LEFT_FACING_ORIENTATION else st_right_group
                    update_st.update(d3_coord1, d3_coord2, EMPTY_VAL) 

    for i in range(N):
        sys.stdout.write(str(len(adj[i])) + "
")

solve()