def calculate_grundy(n, memo):
    if n in memo:
        return memo[n]
    
    if n == 1:
        memo[n] = 0
        return 0
    
    grundy_values = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            grundy_values.add(calculate_grundy(i, memo))
            if i != 1:  # Don't consider n itself (when i = 1, n // i = n)
                grundy_values.add(calculate_grundy(n // i, memo))
    
    g = 0
    while g in grundy_values:
        g += 1
    
    memo[n] = g
    return g

def main():
    n = int(input())
    a = list(map(int, input().split()))
    
    memo = {}
    nim_sum = 0
    for num in a:
        nim_sum ^= calculate_grundy(num, memo)
    
    if nim_sum != 0:
        print("Anna")
    else:
        print("Bruno")

if __name__ == "__main__":
    main()