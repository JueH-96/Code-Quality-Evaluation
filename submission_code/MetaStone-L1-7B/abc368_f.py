def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    max_A = 10**5
    sum_exp = [0] * (max_A + 1)
    
    for x in range(2, max_A + 1):
        temp = x
        total = 0
        
        if temp % 2 == 0:
            count = 0
            while temp % 2 == 0:
                count += 1
                temp //= 2
            total += count
        
        i = 3
        while i * i <= temp:
            if temp % i == 0:
                count = 0
                while temp % i == 0:
                    count += 1
                    temp //= i
                total += count
            i += 2
        
        if temp > 1:
            total += 1
        
        sum_exp[x] = total
    
    n = int(data[0])
    a = list(map(int, data[1:n+1]))
    
    xor_sum = 0
    for num in a:
        xor_sum ^= sum_exp[num]
    
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()