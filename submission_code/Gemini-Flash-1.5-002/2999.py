class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        s1_list = list(s1)
        s2_list = list(s2)
        
        def swap(arr, i, j):
            arr[i], arr[j] = arr[j], arr[i]

        def solve(arr):
            q = [arr]
            visited = {tuple(arr)}
            while q:
                curr = q.pop(0)
                if "".join(curr) == "".join(s2_list):
                    return True
                for i in range(len(curr)):
                    for j in range(i+2, len(curr)):
                        if j - i == 2:
                            next_arr = curr[:]
                            swap(next_arr, i, j)
                            if tuple(next_arr) not in visited:
                                visited.add(tuple(next_arr))
                                q.append(next_arr)
            return False

        return solve(s1_list)