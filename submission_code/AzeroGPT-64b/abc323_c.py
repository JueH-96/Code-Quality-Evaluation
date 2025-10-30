import sys
#input = sys.stdin.buffer.readline

def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    halfway_scores = []
    A_desc = sorted(A, reverse=True)
    for p in range(N):
        NS = input()
        p_score = 0
        p_solved = 0
        for idx, pr in enumerate(NS):
            if pr == "o":
                p_score += A[idx]
                p_solved += 1
        halfway_scores.append([p_score + p + 1, p, p_solved])

    halfway_scores_desc = sorted(halfway_scores, reverse=True)

    for score_info in halfway_scores:
        score1, player, solved = score_info
        need_points = 1
        for other in halfway_scores_desc:
            score2, _, solved2 = other
            if score2 >= score1:
                gap = score2 - (score1 - A_desc[solved] * (need_points - 1))
                if gap == 0:
                    need_points += 1
                elif gap % 100 == 0:
                    need_points += gap // 100
                    break
                else:
                    need_points += (gap // 100) + 1
                    break
        num_left = M - solved
        ans = min(num_left, need_points)
        print(ans)

main()