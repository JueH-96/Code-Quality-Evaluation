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
    
    # Find the maximum prices of main and side dishes
    max_main = max(a)
    max_side = max(b)
    
    # Check if the combination of max_main and max_side is forbidden
    max_main_index = a.index(max_main)
    max_side_index = b.index(max_side)
    
    if (max_main_index, max_side_index) not in forbidden_pairs:
        # If not forbidden, this is the most expensive valid meal
        print(max_main + max_side)
    else:
        # Otherwise, we need to find the next best combination
        # We will consider two cases:
        # 1. max_main with another side dish
        # 2. max_side with another main dish
        
        # Case 1: max_main with another side dish
        max_price = 0
        for j in range(M):
            if j != max_side_index and (max_main_index, j) not in forbidden_pairs:
                max_price = max(max_price, max_main + b[j])
        
        # Case 2: max_side with another main dish
        for i in range(N):
            if i != max_main_index and (i, max_side_index) not in forbidden_pairs:
                max_price = max(max_price, a[i] + max_side)
        
        print(max_price)

if __name__ == "__main__":
    main()