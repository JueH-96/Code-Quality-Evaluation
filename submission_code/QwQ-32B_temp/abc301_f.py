MOD = 998244353

def encode_state(step, first_char):
    if step == 0:
        return 0
    elif step == 1:
        return 1 + (ord(first_char) - ord('A'))
    elif step == 2:
        return 27 + (ord(first_char) - ord('A'))
    elif step == 3:
        return 53 + (ord(first_char) - ord('A'))

def decode_state(s):
    if s == 0:
        return (0, None)
    elif 1 <= s <= 26:
        return (1, chr(ord('A') + (s - 1)))
    elif 27 <= s <= 52:
        return (2, chr(ord('A') + (s - 27)))
    else:  # 53-78
        return (3, chr(ord('A') + (s - 53)))

def get_next_states(s, c):
    step, first_char = decode_state(s)
    next_states = []
    if step == 0:
        if c.isupper():
            next_states.append(0)
            new_s = encode_state(1, c)
            next_states.append(new_s)
        else:
            next_states.append(0)
    elif step == 1:
        f = first_char
        if c.isupper():
            if c == f:
                new_s2 = encode_state(2, f)
                next_states.append(new_s2)
                next_states.append(s)
            else:
                new_s1 = encode_state(1, c)
                next_states.append(new_s1)
                next_states.append(s)
        else:
            next_states.append(s)
    elif step == 2:
        f = first_char
        if c.islower():
            new_s3 = encode_state(3, f)
            next_states.append(new_s3)
            next_states.append(s)
        else:
            new_s1 = encode_state(1, c)
            next_states.append(new_s1)
            next_states.append(s)
    elif step == 3:
        f = first_char
        if c.isupper():
            next_states.append(s)
            new_s1 = encode_state(1, c)
            next_states.append(new_s1)
        else:
            next_states.append(s)
    return next_states

# Precompute transitions for uppercase and lowercase
trans_upper = [dict() for _ in range(79)]
trans_lower = [dict() for _ in range(79)]

for s in range(79):
    # Compute for uppercase
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        next_states = get_next_states(s, c)
        for ns in next_states:
            if ns not in trans_upper[s]:
                trans_upper[s][ns] = 0
            trans_upper[s][ns] += 1
    # Compute for lowercase
    for c in 'abcdefghijklmnopqrstuvwxyz':
        next_states = get_next_states(s, c)
        for ns in next_states:
            if ns not in trans_lower[s]:
                trans_lower[s][ns] = 0
            trans_lower[s][ns] += 1

# Compute trans_wild as the sum of uppercase and lowercase transitions
trans_wild = [dict() for _ in range(79)]
for s in range(79):
    for ns in trans_upper[s]:
        trans_wild[s][ns] = trans_upper[s][ns]
    for ns in trans_lower[s]:
        if ns in trans_wild[s]:
            trans_wild[s][ns] += trans_lower[s][ns]
        else:
            trans_wild[s][ns] = trans_lower[s][ns]

def main():
    S = input().strip()
    current_dp = [0] * 79
    current_dp[0] = 1

    for c in S:
        new_dp = [0] * 79
        if c == '?':
            # Use precomputed wildcard transitions
            for s in range(79):
                if current_dp[s] == 0:
                    continue
                cnt = current_dp[s]
                for ns in trans_wild[s]:
                    new_dp[ns] = (new_dp[ns] + cnt * trans_wild[s][ns]) % MOD
        else:
            # Compute transitions for this specific character
            for s in range(79):
                if current_dp[s] == 0:
                    continue
                cnt = current_dp[s]
                next_states = get_next_states(s, c)
                for ns in next_states:
                    new_dp[ns] = (new_dp[ns] + cnt) % MOD
        current_dp = new_dp

    ans = sum(current_dp) % MOD
    print(ans)

if __name__ == "__main__":
    main()