def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        digits = list(map(int, data[idx:idx+n]))
        idx += n
        
        max_product = 0
        # Try adding 1 to each digit one by one
        for i in range(n):
            digits[i] += 1
            product = 1
            for d in digits:
                product *= d
            if product > max_product:
                max_product = product
            digits[i] -= 1  # revert the change
        
        print(max_product)


# Call solve() after definition
if __name__ == "__main__":
    solve()