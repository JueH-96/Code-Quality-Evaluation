def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # Convert input vertices to 0-indexed representation.
    a = [int(x) - 1 for x in data[1:]]
    
    # dp[i] will store the number of vertices reachable from vertex i (including itself).
    # Initially, dp[i] = -1 indicates the value is not computed.
    dp = [-1] * n

    # Process each vertex that is not computed yet.
    for i in range(n):
        if dp[i] != -1:
            continue
        # pos is used to store the position (index) of each vertex in the current chain (stack)
        # so that we can identify when a cycle occurs.
        pos = {}
        # stack will track the order of vertices we see while following the chain.
        stack = []
        cur = i
        
        while True:
            # If we reached a vertex whose dp is already known, we can unwind the stack.
            if dp[cur] != -1:
                known = dp[cur]
                # Unwind the chain and update dp, adding one at each step.
                for j in range(len(stack) - 1, -1, -1):
                    known += 1
                    dp[stack[j]] = known
                break
                
            # If cur is already in the current chain, we have found a cycle.
            if cur in pos:
                cycle_start = pos[cur]
                cycle_length = len(stack) - cycle_start
                # For vertices in the cycle, the reachable count is the cycle length.
                for idx in range(cycle_start, len(stack)):
                    dp[stack[idx]] = cycle_length
                # Now, for any vertices before the cycle in the chain, add the distance
                # from that vertex to the cycle start plus the cycle length.
                known = dp[stack[cycle_start]]
                for j in range(cycle_start - 1, -1, -1):
                    known += 1
                    dp[stack[j]] = known
                break
                
            # Otherwise, record the current vertex and move on.
            pos[cur] = len(stack)
            stack.append(cur)
            cur = a[cur]
            
    # The result is the sum over all vertices' reachable counts.
    result = sum(dp)
    sys.stdout.write(str(result) + "
")
 
if __name__ == '__main__':
    main()