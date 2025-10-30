import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    S = list(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    stored_value = S[:]
    update_time = [0] * N
    current_time = 0
    last_global_type = 0  # 0: none, 2: lower all uppercase, 3: upper all lowercase
    last_global_time = 0

    for _ in range(Q):
        t = int(data[ptr])
        x = int(data[ptr+1])
        c = data[ptr+2]
        ptr +=3
        if t == 1:
            idx = x - 1
            stored_value[idx] = c
            update_time[idx] = current_time
            current_time +=1
        else:
            last_global_type = t
            last_global_time = current_time
            current_time +=1

    result = []
    if last_global_type == 0:
        result = stored_value
    else:
        for i in range(N):
            if update_time[i] > last_global_time:
                result.append(stored_value[i])
            else:
                c = stored_value[i]
                if last_global_type == 2:
                    # type 2: uppercase to lowercase
                    if c.isupper():
                        result.append(c.lower())
                    else:
                        result.append(c)
                else:
                    # type 3: lowercase to uppercase
                    if c.islower():
                        result.append(c.upper())
                    else:
                        result.append(c)
    print(''.join(result))

if __name__ == "__main__":
    main()