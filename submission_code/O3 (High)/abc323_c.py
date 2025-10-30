import sys

def main() -> None:
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return
    it = iter(input_data)
    N, M = map(int, next(it).split())
    A = list(map(int, next(it).split()))
    S = [next(it).strip() for _ in range(N)]

    # current scores of all players
    current = []
    for i in range(N):
        score = i + 1  # bonus points (players are 1-indexed)
        for j, ch in enumerate(S[i]):
            if ch == 'o':
                score += A[j]
        current.append(score)

    for i in range(N):
        # highest score among the other players
        high = max(current[:i] + current[i + 1:])

        if current[i] > high:
            print(0)
            continue

        need = high - current[i] + 1            # extra points required to exceed
        candidates = [A[j] for j, ch in enumerate(S[i]) if ch == 'x']
        candidates.sort(reverse=True)           # take biggest first

        total = cnt = 0
        for v in candidates:
            total += v
            cnt += 1
            if total >= need:
                print(cnt)
                break

if __name__ == "__main__":
    main()