# YOUR CODE HERE
def find_palindrome(N):
    if N < 10:
        return str(N)
    
    for length in range(1, 1001):
        for first_digit in range(1, 10):
            for middle in range(10):
                if length == 1:
                    candidate = str(first_digit)
                elif length == 2:
                    candidate = f"{first_digit}{first_digit}"
                else:
                    half_length = (length - 1) // 2
                    candidate = f"{first_digit}{'*' * half_length}{middle}{'*' * half_length}{first_digit}"
                
                try:
                    if eval(candidate) == N:
                        return candidate
                except:
                    continue
    
    return -1

N = int(input())
print(find_palindrome(N))