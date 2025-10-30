# YOUR CODE HERE
def solve():
    n = int(input())
    
    def is_palindrome(s):
        return s == s[::-1]

    def evaluate(s):
        try:
            return eval(s)
        except:
            return -1

    for length in range(1, 1001):
        if length == 1:
            s = str(n)
            if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                print(s)
                return
        elif length == 2:
            for i in range(1, 10):
                s = str(i) * 2
                if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                    print(s)
                    return
        elif length == 3:
            for i in range(1, 10):
                for j in '123456789*':
                    s = str(i) + j + str(i)
                    if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                        if evaluate(s) == n:
                            print(s)
                            return
        elif length == 5:
            for i in range(1, 10):
                for j in '123456789*':
                    for k in '123456789*':
                        s = str(i) + j + k + j + str(i)
                        if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                            if evaluate(s) == n:
                                print(s)
                                return
        elif length % 2 != 0:
            for i in range(1, 10):
                for j in range(1, 100000):
                    s = str(i) + '*' + str(j) + '*' + str(n // (i * j)) + '*' + str(j) + '*' + str(i)
                    if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                        if evaluate(s) == n:
                            print(s)
                            return
        else:
            for i in range(1, 100000):
                for j in range(1, 100000):
                    s = str(i) + '*' + str(j) + '*' + str(n // (i*j)) + '*' + str(j) + '*' + str(i)
                    if 1 <= len(s) <= 1000 and is_palindrome(s) and '0' not in s and all(c in '123456789*' for c in s) and s[0] in '123456789' and evaluate(s) == n:
                        if evaluate(s) == n:
                            print(s)
                            return
    print("-1")

solve()