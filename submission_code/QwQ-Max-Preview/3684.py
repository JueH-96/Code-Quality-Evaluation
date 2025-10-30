class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        split_p = p.split('*')
        left, right = split_p[0], split_p[1]
        left_len, right_len = len(left), len(right)
        
        if left_len + right_len == 0:
            return False
        
        s_len = len(s)
        if left_len + right_len > s_len:
            return False
        
        # Collect indices where left is a prefix
        left_indices = []
        for i in range(s_len - left_len + 1):
            if s[i:i+left_len] == left:
                left_indices.append(i)
        
        # Collect indices where right is a suffix
        right_indices = []
        for j in range(s_len - right_len + 1):
            if s[j:j+right_len] == right:
                right_indices.append(j)
        
        # Check for valid pairs of i and j
        for i in left_indices:
            j_min = i + left_len
            j_max = s_len - right_len
            for j in right_indices:
                if j_min <= j <= j_max:
                    return True
        
        return False