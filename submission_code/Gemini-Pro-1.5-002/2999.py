class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        s1_list = list(s1)
        s2_list = list(s2)

        def find_permutations(arr):
            permutations = set()
            
            def generate_permutations(current_arr):
                permutations.add(tuple(current_arr))
                
                for i in range(len(current_arr)):
                    if i + 2 < len(current_arr):
                        new_arr = current_arr[:]
                        new_arr[i], new_arr[i+2] = new_arr[i+2], new_arr[i]
                        generate_permutations(new_arr)
            
            generate_permutations(arr)
            return permutations

        s1_permutations = find_permutations(s1_list)
        
        for perm in s1_permutations:
            if list(perm) == s2_list:
                return True
        
        return False