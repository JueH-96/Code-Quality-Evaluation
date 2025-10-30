def min_operations_to_remove_black_cells(s, k):
    n = len(s)
    paper = list(s)  # Convert the string to a list for easy manipulation
    operations = 0
    
    for i in range(n):
        if paper[i] == 'B':  # If the cell is black
            operations += 1
            for j in range(i, min(i + k, n)):
                paper[j] = 'W'  # Make the next k cells white
    
    return operations

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    print(min_operations_to_remove_black_cells(s, k))