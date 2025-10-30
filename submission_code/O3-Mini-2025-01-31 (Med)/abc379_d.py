def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return

    Q = int(data[0])
    # Global "inc" is the cumulative days waited.
    inc = 0
    # For each plant, we store its "creation time" which is the value of inc when it was planted.
    # Since type2 queries only increase inc, the plants are inserted in non-decreasing order.
    plants = deque()

    output = []
    pos = 1
    for _ in range(Q):
        query_type = data[pos]
        pos += 1

        if query_type == "1":
            # Plant a new flower with height 0; its effective creation time is current inc.
            plants.append(inc)

        elif query_type == "2":
            # Wait for T days.
            T = int(data[pos])
            pos += 1
            inc += T

        elif query_type == "3":
            # Harvest all plants with height >= H.
            # Actual height of a plant = current inc - creation time.
            # We want plants with: inc - creation_time >= H  => creation_time <= inc - H.
            H = int(data[pos])
            pos += 1
            threshold = inc - H
            cnt = 0
            # Since plants are stored in non-decreasing order of their creation time,
            # we pop from the left until the condition is not met.
            while plants and plants[0] <= threshold:
                plants.popleft()
                cnt += 1
            output.append(str(cnt))

    sys.stdout.write("
".join(output))


if __name__ == '__main__':
    main()