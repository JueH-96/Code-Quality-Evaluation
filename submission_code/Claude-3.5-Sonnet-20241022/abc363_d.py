def get_nth_palindrome(n):
    if n == 1:
        return 0
    n -= 1  # Account for 0 being first palindrome
    
    # For length L, count of palindromes = 9*10^((L-1)//2) for L>1
    length = 1
    count = 0
    while True:
        # Count palindromes of current length
        if length == 1:
            curr_count = 9
        else:
            curr_count = 9 * pow(10, (length-1)//2)
            
        if count + curr_count >= n:
            break
        count += curr_count
        length += 1
    
    # n is now relative to palindromes of length 'length'
    n -= count
    
    # For length L, first number is 10^(L-1)
    if length == 1:
        return n
    
    # Calculate the prefix
    mid = (length-1)//2
    start = pow(10, mid)
    prefix = start + n-1
    
    # Construct full palindrome
    num = prefix
    if length % 2 == 0:
        temp = prefix
    else:
        temp = prefix//10
        
    # Add reverse of prefix
    while temp > 0:
        num = num * 10 + temp % 10
        temp //= 10
        
    return num

n = int(input())
print(get_nth_palindrome(n))