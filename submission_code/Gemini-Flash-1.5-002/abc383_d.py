def count_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i * i != n:
                count += 1
        i += 1
    return count

def count_numbers_with_9_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 2):
        num_div = count_divisors(i)
        if num_div == 9:
            
            j = 1
            while i**j <= n:
                count +=1
                j+=1

    
    
    return count


n = int(input())
print(count_numbers_with_9_divisors(n))