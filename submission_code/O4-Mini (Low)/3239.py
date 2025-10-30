from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If already equal, no operations needed.
        if x == y:
            return 0
        
        # We'll BFS over the space of possible x-values.
        # To allow for increments beyond max(x,y) to enable a division later,
        # we set an upper bound.
        bound = max(x, y) * 11 + 10
        
        # visited[i] will be True once we've enqueued value i.
        visited = [False] * (bound + 1)
        dq = deque()
        
        # Start from x, with 0 steps taken.
        dq.append((x, 0))
        visited[x] = True
        
        while dq:
            cur, steps = dq.popleft()
            
            # Try decrement
            if cur - 1 >= 1 and not visited[cur - 1]:
                if cur - 1 == y:
                    return steps + 1
                visited[cur - 1] = True
                dq.append((cur - 1, steps + 1))
            
            # Try increment
            if cur + 1 <= bound and not visited[cur + 1]:
                if cur + 1 == y:
                    return steps + 1
                visited[cur + 1] = True
                dq.append((cur + 1, steps + 1))
            
            # Try divide by 5
            if cur % 5 == 0:
                nxt = cur // 5
                if nxt >= 1 and not visited[nxt]:
                    if nxt == y:
                        return steps + 1
                    visited[nxt] = True
                    dq.append((nxt, steps + 1))
            
            # Try divide by 11
            if cur % 11 == 0:
                nxt = cur // 11
                if nxt >= 1 and not visited[nxt]:
                    if nxt == y:
                        return steps + 1
                    visited[nxt] = True
                    dq.append((nxt, steps + 1))
        
        # By problem statement, we should always be able to reach y.
        # But just in case, return some sentinel.
        return -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumOperationsToMakeEqual(26, 1))  # Expect 3
    print(sol.minimumOperationsToMakeEqual(54, 2))  # Expect 4
    print(sol.minimumOperationsToMakeEqual(25, 30)) # Expect 5