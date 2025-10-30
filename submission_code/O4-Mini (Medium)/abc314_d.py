import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().rstrip('
')
    Q = int(input().strip())

    # current characters
    chars = list(S)
    # timestamp of last individual update for each position
    last_upd = [0] * N
    # timestamp and type of last mass operation
    last_mass_time = 0
    last_mass_type = 0  # 0 = none, 2 = to lowercase, 3 = to uppercase

    for t in range(1, Q+1):
        line = input().split()
        ti = int(line[0])
        if ti == 1:
            xi = int(line[1]) - 1
            ci = line[2]
            chars[xi] = ci
            last_upd[xi] = t
        else:
            # mass operation
            last_mass_time = t
            last_mass_type = ti

    # build final string
    out = []
    if last_mass_type == 2:
        # lowercase all unless individually updated after mass
        for i in range(N):
            if last_upd[i] > last_mass_time:
                out.append(chars[i])
            else:
                out.append(chars[i].lower())
    elif last_mass_type == 3:
        # uppercase all unless individually updated after mass
        for i in range(N):
            if last_upd[i] > last_mass_time:
                out.append(chars[i])
            else:
                out.append(chars[i].upper())
    else:
        # no mass operation at all
        out = chars

    sys.stdout.write(''.join(out))

if __name__ == "__main__":
    main()