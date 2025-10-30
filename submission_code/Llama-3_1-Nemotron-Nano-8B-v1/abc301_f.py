mod = 998244353

S = input().strip()

state0 = 1
state1 = [0] * 26
state2 = [0] * 26
state3 = [0] * 26
state4 = 0

for c in S:
    new_state0 = 0
    new_state1 = [0] * 26
    new_state2 = [0] * 26
    new_state3 = [0] * 26
    new_state4 = 0

    if c == '?':
        possible_chars = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    else:
        possible_chars = [c]

    for char in possible_chars:
        if char.isupper():
            idx = ord(char) - ord('A')
            new_state1[idx] = (new_state1[idx] + state0) % mod
        else:
            new_state0 = (new_state0 + state0) % mod

        for a in range(26):
            current_a = state1[a]
            if current_a == 0:
                continue
            if char.isupper():
                a_idx = ord(char) - ord('A')
                if a_idx == a:
                    new_state2[a] = (new_state2[a] + current_a) % mod
                else:
                    new_state1[a] = (new_state1[a] + current_a) % mod
                    new_state1[a_idx] = (new_state1[a_idx] + current_a) % mod
            else:
                if chr(a + ord('A')).lower() == char:
                    new_state2[a] = (new_state2[a] + current_a) % mod
                else:
                    new_state1[a] = (new_state1[a] + current_a) % mod

        for a in range(26):
            current_a = state2[a]
            if current_a == 0:
                continue
            if char.islower() and char == chr(a + ord('A')).lower():
                new_state3[a] = (new_state3[a] + current_a) % mod
            else:
                new_state2[a] = (new_state2[a] + current_a) % mod

        for a in range(26):
            current_a = state3[a]
            if current_a == 0:
                continue
            if char.isupper() and char == chr(a + ord('A')):
                new_state4 = (new_state4 + current_a) % mod
            else:
                new_state3[a] = (new_state3[a] + current_a) % mod

        new_state4 = (new_state4 + state4) % mod

    state0 = new_state0 % mod
    for i in range(26):
        state1[i] = new_state1[i] % mod
    for i in range(26):
        state2[i] = new_state2[i] % mod
    for i in range(26):
        state3[i] = new_state3[i] % mod
    state4 = new_state4 % mod

q = S.count('?')
total = pow(52, q, mod)
answer = (total - state4) % mod
print(answer)