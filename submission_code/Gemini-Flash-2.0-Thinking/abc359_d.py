def solve():
    n, k = map(int, input().split())
    s = input()
    q_indices = [i for i, char in enumerate(s) if char == '?']
    q_count = len(q_indices)
    mod = 998244353
    good_count = 0

    def is_palindrome(sub):
        return sub == sub[::-1]

    def is_good(string):
        for i in range(len(string) - k + 1):
            substring = string[i:i+k]
            if is_palindrome(substring):
                return False
        return True

    for i in range(2**q_count):
        temp_s = list(s)
        temp_val = i
        for j in range(q_count):
            if (temp_val % 2) == 0:
                temp_s[q_indices[j]] = 'A'
            else:
                temp_s[q_indices[j]] = 'B'
            temp_val //= 2

        generated_string = "".join(temp_s)
        if is_good(generated_string):
            good_count = (good_count + 1) % mod

    print(good_count)

solve()