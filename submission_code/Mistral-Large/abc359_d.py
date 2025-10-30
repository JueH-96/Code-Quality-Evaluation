import sys

MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    q_count = S.count('?')
    total_combinations = 2 ** q_count
    good_strings_count = 0

    def generate_combinations(index, current_combination):
        nonlocal good_strings_count
        if index == q_count:
            candidate = list(S)
            comb_index = 0
            for i in range(N):
                if candidate[i] == '?':
                    candidate[i] = current_combination[comb_index]
                    comb_index += 1
            candidate = ''.join(candidate)
            if all(not is_palindrome(candidate[i:i+K]) for i in range(N-K+1)):
                good_strings_count = (good_strings_count + 1) % MOD
            return

        for char in ['A', 'B']:
            generate_combinations(index + 1, current_combination + [char])

    generate_combinations(0, [])
    return good_strings_count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    S = data[2]

    result = count_good_strings(N, K, S)
    print(result)

if __name__ == "__main__":
    main()