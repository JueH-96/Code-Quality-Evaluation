class Solution:
    def maxOperations(self, s: str) -> int:
        s_list = list(s)
        operations = 0
        while True:
            found_operation = False
            for i in range(len(s_list) - 1):
                if s_list[i] == '1' and s_list[i+1] == '0':
                    char_to_move = s_list[i]
                    s_list[i] = '0'
                    j = i + 1
                    while j < len(s_list) and s_list[j] == '0':
                        j += 1
                    s_list.insert(j, char_to_move)
                    del s_list[i]
                    operations += 1
                    found_operation = True
                    break
            if not found_operation:
                break
        return operations