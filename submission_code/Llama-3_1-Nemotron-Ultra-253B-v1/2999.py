class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        def get_perms(s):
            perms = set()
            # Original
            perms.add(s)
            # Swap 0 and 2
            lst = list(s)
            lst[0], lst[2] = lst[2], lst[0]
            perms.add(''.join(lst))
            # Swap 1 and 3
            lst = list(s)
            lst[1], lst[3] = lst[3], lst[1]
            perms.add(''.join(lst))
            # Swap both 0-2 and 1-3
            lst = list(s)
            lst[0], lst[2] = lst[2], lst[0]
            lst[1], lst[3] = lst[3], lst[1]
            perms.add(''.join(lst))
            return perms
        
        return len(get_perms(s1) & get_perms(s2)) > 0