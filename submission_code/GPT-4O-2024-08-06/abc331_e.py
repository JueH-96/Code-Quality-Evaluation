# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    L = int(data[index])
    index += 1
    
    a = list(map(int, data[index:index+N]))
    index += N
    b = list(map(int, data[index:index+M]))
    index += M
    
    forbidden_pairs = set()
    for _ in range(L):
        c = int(data[index]) - 1
        index += 1
        d = int(data[index]) - 1
        index += 1
        forbidden_pairs.add((c, d))
    
    # Find the maximum values
    max_a = max(a)
    max_b = max(b)
    
    # Find indices of maximum values
    max_a_indices = [i for i in range(N) if a[i] == max_a]
    max_b_indices = [j for j in range(M) if b[j] == max_b]
    
    # Check if the combination of max_a and max_b is forbidden
    max_price = max_a + max_b
    is_forbidden = False
    
    for i in max_a_indices:
        for j in max_b_indices:
            if (i, j) in forbidden_pairs:
                is_forbidden = True
                break
        if is_forbidden:
            break
    
    if not is_forbidden:
        print(max_price)
        return
    
    # If the max combination is forbidden, find the next best option
    # Try max_a with other side dishes
    second_best_price = 0
    for j in range(M):
        if (max_a_indices[0], j) not in forbidden_pairs:
            second_best_price = max(second_best_price, max_a + b[j])
    
    # Try max_b with other main dishes
    for i in range(N):
        if (i, max_b_indices[0]) not in forbidden_pairs:
            second_best_price = max(second_best_price, a[i] + max_b)
    
    print(second_best_price)