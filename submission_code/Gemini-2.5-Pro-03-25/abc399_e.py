# YOUR CODE HERE
import sys

# Setting a reasonable recursion depth limit, just in case. Default is usually 1000.
# Max path length in the character graph is 26, so default limit should be sufficient.
# sys.setrecursionlimit(2000) 

def solve():
    """
    Solves the problem of finding the minimum operations to transform string S to T.
    The allowed operation is: Choose two lowercase English letters x, y and replace 
    every occurrence of x in S with y.
    """
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # If S and T are already identical, no operations are needed.
    if S == T:
        print(0)
        return

    # target_map stores the required transformation c -> target_map[c] for each character c in S.
    target_map = {} 
    possible = True
    # chars_in_S keeps track of unique characters present in the initial string S.
    chars_in_S = set() 

    # First pass over the strings:
    # 1. Check if the required transformations are consistent. 
    #    If S[i] == S[j], then T[i] must equal T[j] for the transformation to be possible.
    # 2. Build the target map based on pairs (S[i], T[i]).
    # 3. Collect all unique characters present in S.
    for i in range(N):
        s_char = S[i]
        t_char = T[i]
        chars_in_S.add(s_char)
        
        if s_char in target_map:
            # If s_char already has a required target, check if the current T[i] is consistent.
            if target_map[s_char] != t_char:
                possible = False
                break
        else:
            # If s_char doesn't have a target yet, assign T[i] as its target.
            target_map[s_char] = t_char

    # If any inconsistency was found, it's impossible to transform S to T.
    if not possible:
        print("-1")
        return

    # Ensure all characters from S are keys in target_map.
    # If a character `s_char` appeared in S, but for all its occurrences `S[i] == T[i]`,
    # it wouldn't have been added to `target_map` in the loop above.
    # We add mapping `s_char -> s_char` for such cases to complete the map.
    # This does not affect `V_active` calculation but clarifies the structure.
    for s_char_in_s in chars_in_S:
         if s_char_in_s not in target_map:
             target_map[s_char_in_s] = s_char_in_s

    # V_active: set of characters `c` such that `c` appears in S and needs transformation
    # (i.e., its target character target_map[c] is different from c itself).
    V_active = set()
    for s_char in chars_in_S:
        if s_char in target_map and target_map[s_char] != s_char:
             V_active.add(s_char)

    # The base number of operations required is the count of characters that need to change.
    # Each character `c` in V_active requires at least one operation involving it.
    total_ops = len(V_active)
    
    # We use Depth First Search (DFS) to analyze the structure of transformations graph.
    # The graph nodes are characters 'a' through 'z'. An edge exists from `u` to `v` if `target_map[u] == v` and `u != v`.
    # We need to detect cycles in this graph. Cycles of length K > 1 require an extra operation.
    
    # State tracking for DFS: 0=WHITE (unvisited), 1=GRAY (visiting), 2=BLACK (visited and finished)
    state = {} 
    for char_code in range(ord('a'), ord('z') + 1):
        state[chr(char_code)] = 0 # Initialize all characters state to WHITE

    # path_stack keeps track of nodes in the current DFS path. Used to identify nodes in a cycle.
    path_stack = [] 
    # on_stack set allows O(1) check if a node is currently in the recursion stack path.
    on_stack = set() 

    # Shared mutable state for DFS to update cycle bonus and track nodes part of counted cycles.
    # We use a dictionary to pass this mutable state through recursion easily.
    dfs_shared = {
        'cycle_bonus': 0, # Counts the number of cycles of length K > 1 found.
        'counted_cycle_nodes': set() # Stores nodes belonging to cycles for which bonus is already added.
    }

    def dfs(u):
        """ Performs DFS starting from node u to detect cycles. """
        # Mark current node as visiting (GRAY)
        state[u] = 1 
        path_stack.append(u)
        on_stack.add(u)

        # Check if u requires transformation (maps to some v where u != v)
        if u in target_map and target_map[u] != u:
            v = target_map[u] # The neighbor node v is the target of u
            
            if v in on_stack: # Cycle detected: Found a back edge to node 'v' which is currently on the recursion stack.
                
                # Check if this cycle's bonus has already been counted. We check using node 'v'.
                # If 'v' is not in counted_cycle_nodes, this cycle is new.
                if v not in dfs_shared['counted_cycle_nodes']:
                    try:
                        # Find the start index of the cycle in the path stack to identify all nodes in the cycle.
                        idx = path_stack.index(v) 
                        cycle_nodes = path_stack[idx:]
                        K = len(cycle_nodes)
                        # Add bonus +1 only for cycles of length K > 1. Self-loops (K=1) don't need extra op.
                        if K > 1: 
                           dfs_shared['cycle_bonus'] += 1
                           # Mark all nodes in this cycle as counted to prevent double counting this cycle bonus.
                           for node in cycle_nodes:
                               dfs_shared['counted_cycle_nodes'].add(node)
                    except ValueError:
                         # This block should theoretically not be reached if v is in on_stack.
                         # Included as a safeguard.
                         pass
            
            elif state[v] == 0: # If neighbor v is WHITE (unvisited)
                # Recursively call DFS on the neighbor v.
                dfs(v)
            # Else: neighbor v is BLACK (state[v] == 2). This means v and its reachable subgraph
            # have already been fully explored. We stop exploring this path further.

        # Finished exploring from u. Mark as fully visited (BLACK).
        state[u] = 2 
        
        # Backtrack step: remove u from path stack and on_stack set.
        # These checks ensure robustness although should ideally not fail if logic is correct.
        if path_stack:
             path_stack.pop()
        if u in on_stack:
             on_stack.remove(u)


    # Iterate through all characters 'a' through 'z'.
    for char_code in range(ord('a'), ord('z') + 1):
         curr_char = chr(char_code)
         # Start DFS only for characters that:
         # 1. Are in V_active (require transformation).
         # 2. Haven't been visited yet (are WHITE).
         if curr_char in V_active and state[curr_char] == 0: 
             # This starts exploration of a new connected component in the transformation graph.
             dfs(curr_char)
    
    # The final minimum number of operations is the sum of base operations (one per changing character)
    # plus the cycle bonuses (one extra per cycle of length > 1).
    print(total_ops + dfs_shared['cycle_bonus'])

# Execute the solver function
solve()