def solve():
    n = int(input())
    
    def generate_carpet(level):
        if level == 0:
            return ["#"]
        lower_level_carpet = generate_carpet(level - 1)
        block_size = 3**(level - 1)
        carpet_rows = []
        for block_row_index in range(3):
            for local_row_index in range(block_size):
                current_row_string = ""
                for block_col_index in range(3):
                    if block_row_index == 1 and block_col_index == 1:
                        current_row_string += "." * block_size
                    else:
                        current_row_string += lower_level_carpet[local_row_index]
                carpet_rows.append(current_row_string)
        return carpet_rows
        
    result_carpet = generate_carpet(n)
    for row_str in result_carpet:
        print(row_str)

if __name__ == '__main__':
    solve()