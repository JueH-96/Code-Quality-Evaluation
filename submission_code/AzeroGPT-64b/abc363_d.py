n = int(input())
def smallest_palindrome(n):
    def next_palindrome(n):
        s = str(n)
        length = len(s)
        if length % 2 == 0: # even
            half = int(s[:length//2])
            while True:
                incremented = s[:length // 2] + str(half)[::-1]
                if int(incremented) > n:
                    yield incremented
                half += 1
        else: # odd
            half = int(s[:(length-1)//2])
            while True:
                incremented = s[:(length-1)//2] + s[length//2] + str(half)[::-1]
                if int(incremented) > n:
                    yield incremented
                half += 1
    result = 0
    while n > 0:
        result += 1
        n -= 1
        for p in next_palindrome(result):
            result = int(p)
            break
    return result
print(smallest_palindrome(n))