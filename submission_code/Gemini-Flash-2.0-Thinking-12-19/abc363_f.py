def generate_palindrome_numbers_up_to(max_val):
    palindromes = []
    for length in range(1, 15):
        def generate_first_half(current_half):
            if len(current_half) == (length + 1) // 2:
                s = current_half
                if length % 2 == 1:
                    first_half = s
                    second_half = s[:-1][::-1]
                else:
                    first_half = s
                    second_half = s[::-1]
                palindrome_str = first_half + second_half
                palindrome_int = int(palindrome_str)
                if palindrome_int <= max_val:
                    palindromes.append(palindrome_int)
                return

            for digit in range(1, 10):
                generate_first_half(current_half + str(digit))

        generate_first_half("")
    palindromes.sort()
    return palindromes

palindromes = generate_palindrome_numbers_up_to(10**12)

def solve(n, factors):
    if n == 1:
        if not factors:
            return None
        return "*".join(factors)
    for p in palindromes:
        if p > n:
            break
        if n % p == 0:
            res = solve(n // p, factors + [str(p)])
            if res is not None:
                s = res
                if len(s) <= 1000:
                    if not s:
                        return None
                    if not s[0].isdigit():
                        return None
                    temp_s_no_star = "".join(x for x in s if x != '*')
                    if temp_s_no_star != temp_s_no_star[::-1]:
                        return None
                    try:
                        if eval(s) == int(eval(s)): # check no float issue
                            return s
                        else:
                            return None
                    except:
                        return None
    return None

n = int(input())
if n <= 0:
    print("-1")
else:
    ans = solve(n, [])
    if ans is None:
        s_n = str(n)
        if s_n.isdigit() and all(c in '123456789' for c in s_n) and s_n == s_n[::-1] and len(s_n) <= 1000:
            print(s_n)
        else:
            print("-1")
    else:
        print(ans)