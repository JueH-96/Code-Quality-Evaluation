def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    
    # Keeps track of whether a male has been born in each family
    has_male = [False] * N
    
    # Index in data after reading N and M
    idx = 2
    
    for _ in range(M):
        family = int(data[idx]); idx += 1
        gender = data[idx]; idx += 1
        
        # Convert family index to 0-based
        family_idx = family - 1
        
        if gender == 'M' and not has_male[family_idx]:
            print("Yes")
            has_male[family_idx] = True
        else:
            print("No")

# DO NOT forget to call main()
main()