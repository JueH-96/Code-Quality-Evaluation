def is_valid_single(s):
    return '0' not in s and s == s[::-1]

def find_two_factor(N):
    max_a = int(N**0.5) + 1
    for a in range(1, max_a + 1):
        if N % a != 0:
            continue
        b = N // a
        a_str = str(a)
        if '0' in a_str:
            continue
        rev_a_str = a_str[::-1]
        rev_a = int(rev_a_str)
        if rev_a == b:
            return f"{a_str}*{rev_a_str}"
    return None

def find_three_factor(N):
    max_a = int(N ** (1/3)) + 1
    for a in range(1, max_a + 1):
        if N % a != 0:
            continue
        a_str = str(a)
        if '0' in a_str:
            continue
        rev_a_str = a_str[::-1]
        rev_a = int(rev_a_str)
        product_a_rev = a * rev_a
        if product_a_rev == 0:
            continue
        if N % product_a_rev != 0:
            continue
        remaining = N // product_a_rev
        remaining_str = str(remaining)
        if '0' in remaining_str:
            continue
        if remaining_str != remaining_str[::-1]:
            continue
        return f"{a_str}*{remaining_str}*{rev_a_str}"
    return None

def find_four_factor(N):
    max_a = int(N**0.5) + 1
    for a in range(1, max_a + 1):
        if N % a != 0:
            continue
        a_str = str(a)
        if '0' in a_str:
            continue
        rev_a_str = a_str[::-1]
        rev_a = int(rev_a_str)
        product_a_rev = a * rev_a
        if product_a_rev == 0:
            continue
        remaining = N // product_a_rev
        two_result = find_two_factor(remaining)
        if two_result:
            parts = two_result.split('*')
            b_str = parts[0]
            rev_b_str = parts[1]
            return f"{a_str}*{b_str}*{rev_b_str}*{rev_a_str}"
    return None

def find_five_factor(N):
    max_a = int(N ** (1/5)) + 1
    for a in range(1, max_a + 1):
        if N % a != 0:
            continue
        a_str = str(a)
        if '0' in a_str:
            continue
        rev_a_str = a_str[::-1]
        rev_a = int(rev_a_str)
        product_a_rev = a * rev_a
        if product_a_rev == 0:
            continue
        if N % product_a_rev != 0:
            continue
        remaining = N // product_a_rev
        three_result = find_three_factor(remaining)
        if three_result:
            parts = three_result.split('*')
            b_str = parts[0]
            c_str = parts[1]
            rev_b_str = parts[2]
            if rev_b_str != b_str[::-1]:
                continue
            return f"{a_str}*{b_str}*{c_str}*{rev_b_str}*{rev_a_str}"
    return None

def main():
    N = int(input().strip())
    n_str = str(N)
    if is_valid_single(n_str):
        print(n_str)
        return
    two_result = find_two_factor(N)
    if two_result is not None:
        print(two_result)
        return
    three_result = find_three_factor(N)
    if three_result is not None:
        print(three_result)
        return
    four_result = find_four_factor(N)
    if four_result is not None:
        print(four_result)
        return
    five_result = find_five_factor(N)
    if five_result is not None:
        print(five_result)
        return
    print(-1)

if __name__ == "__main__":
    main()