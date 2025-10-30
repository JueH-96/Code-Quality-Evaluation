def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Prepare an iterator for inputs.
    it = iter(data)
    N = int(next(it))
    s = list(next(it).decode())
    Q = int(next(it))
    
    # We'll maintain a version for each position indicating whether it has been updated
    # after the most recent global conversion.
    version = [0] * N
    global_version = 0
    # current_global keeps track of the last global operation:
    # 'l' means convert all to lowercase, 'u' means convert all to uppercase.
    current_global = None

    for _ in range(Q):
        t = int(next(it))
        x = int(next(it))
        c = next(it).decode()
        if t == 1:
            # type 1: update the x-th (1-indexed) character to c.
            idx = x - 1
            # No matter the pending conversion, we directly replace the value.
            s[idx] = c
            # Record that this update happened after the current global operation.
            version[idx] = global_version
        elif t == 2:
            # type 2: convert all uppercase letters in S to lowercase.
            global_version += 1
            current_global = 'l'
        elif t == 3:
            # type 3: convert all lowercase letters in S to uppercase.
            global_version += 1
            current_global = 'u'

    # After processing all operations, some elements of s might not have been updated
    # after the latest global conversion. For those, we perform the conversion now.
    if current_global is None:
        # No global conversion was performed.
        output = "".join(s)
    else:
        result = []
        if current_global == 'l':
            for i in range(N):
                # If version[i] < global_version, then the position was not modified
                # after the last global conversion, so we convert it.
                if version[i] < global_version:
                    result.append(s[i].lower())
                else:
                    result.append(s[i])
        else:  # current_global == 'u'
            for i in range(N):
                if version[i] < global_version:
                    result.append(s[i].upper())
                else:
                    result.append(s[i])
        output = "".join(result)

    sys.stdout.write(output)

if __name__ == '__main__':
    main()