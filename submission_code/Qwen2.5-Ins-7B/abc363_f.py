# N = int(input())
def find_palindrome(N):
    if N == 1:
        return "1"
    for i in range(1, 10):
        for j in range(1, 10):
            num = int(f"{i}{j}{i}")
            if num > N:
                continue
            if N % num == 0:
                quotient = N // num
                quotient_str = str(quotient)
                if all(digit in "123456789" for digit in quotient_str) and len(quotient_str) <= 9:
                    return f"{i}{j}{i}*{quotient_str}*{i}{j}{i}"
    return "-1"

N = int(input())
print(find_palindrome(N))