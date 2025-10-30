class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        i = 0
        while i < len(s):
            if s[i].isdigit():
                left_index = i -1
                right_index = i + 1
                
                closest_index = -1
                min_dist = float('inf')

                while left_index >=0 and not s[left_index].isdigit():
                    if i - left_index < min_dist:
                        min_dist = i - left_index
                        closest_index = left_index
                    left_index -=1

                while right_index < len(s) and not s[right_index].isdigit():
                    if right_index - i < min_dist:
                        min_dist = right_index - i
                        closest_index = right_index
                    right_index += 1

                if closest_index != -1:
                    s.pop(i)
                    s.pop(closest_index)
                    i = 0
                else:
                    i+=1
            else:
                i+=1
        return "".join(s)