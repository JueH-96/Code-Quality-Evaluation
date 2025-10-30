class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        s_list = list(s)
        n = len(s_list)

        while True:
            found_operation = False
            for i in range(len(s_list) - 1):
                if s_list[i] == '1' and s_list[i + 1] == '0':
                    operations += 1
                    one_val = s_list[i]
                    del s_list[i]

                    insert_index = i
                    while insert_index < len(s_list) and s_list[insert_index] == '0':
                        insert_index += 1
                    s_list.insert(insert_index, one_val)

                    found_operation = True
                    break
            if not found_operation:
                break
        return operations