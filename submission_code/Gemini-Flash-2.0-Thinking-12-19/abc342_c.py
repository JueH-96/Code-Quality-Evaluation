def solve():
    n = int(input())
    s = input()
    q = int(input())
    operations = []
    for _ in range(q):
        line = input().split()
        operations.append((line[0], line[1]))
    
    s_list = list(s)
    
    for operation in operations:
        char_to_replace, replacement_char = operation
        for i in range(len(s_list)):
            if s_list[i] == char_to_replace:
                s_list[i] = replacement_char
                
    result_string = "".join(s_list)
    print(result_string)

if __name__ == '__main__':
    solve()