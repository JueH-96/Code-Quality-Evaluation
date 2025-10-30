from collections import defaultdict

def main() -> None:
    from sys import stdin
    n = int(stdin.readline())
    pq = [(int(x) - 1 for x in stdin.readline().split())
          for _ in range(n - 1)]
    tree_from = defaultdict(list)
    tree_to = defaultdict(list)
    for p, q in pq:
        tree_from[p].append(q)
        tree_to[q].append(p)
    q_tf, q_tn, q_t_album, q_paper, q_answer = [defaultdict(list) for _ in range(5)]
    for p, q in pq:
        q_tf[p].append(0)
        q_tn[p].append(q)
        q_t_album[p].append(0)
        q_paper[p].append(None)
        q_answer[p].append(None)
    for i in range(n):
        q_tf[i][0] = i
        q_tn[i][0] = i
        q_t_album[i][0] = 1
        q_paper[i][0] = 1
        # q_answer[i][0] = 0 No accepted answer, question is not solved.
    # print(q_tn, q_tf)

    def dfs(x, f, tree=tree_to, q_answer=q_answer, q_tn=q_tn, q_t_album=q_t_album,
            q_paper=q_paper, q_tf=q_tf) -> int:
        if len(q_answer[x][f]) != None:
            return q_answer[x][f]
        if len(q_tn[x][f:]) == 0:
            q_paper[x][f] = 1
            q_answer[x][f] = 0
            return 0
        sub_answers = []
        for c in q_tn[x][f:]:
            v_c = dfs(c, 0, tree=tree, q_answer=q_answer, q_tn=q_tn, q_t_album=q_t_album,
                      q_paper=q_paper, q_tf=q_tf)
            sub_answers.append(v_c)
        sub_answer = sub_answers[0]
        for v_c in sub_answers[1:]:
            sub_answer = (sub_answer * pow(+v_c, -1, MOD)
                          ) % MOD * (v_c + q_t_album[x][f] * pow(1 + v_c, -1, MOD)) % MOD
        q_paper[x][f] = 0x1FFFFFFF & (1 << len(q_tn[x][f:]))
        q_answer[x][f] = (q_paper[x][f] + q_answer[x][f + len(q_tn[x][f:])]
                          ) % MOD * sub_answer % MOD
        q_t_album[x][f] = (sub_answer * (1 << len(q_tn[x][f:])) + q_t_album[x][f + len(q_tn[x][f:])]
                           ) % MOD

        for (u, d) in zip(q_tn[x][f:], q_tf[x][f + 1:]):
            n_answer = dfs(x, f + 1, tree_to, q_answer, q_tn, q_t_album,
                           q_paper, q_tf) * MOD_INV * pow(1 + q_answer[u][0], -1, MOD) % MOD
            q_answer[u][d] = (n_answer - q_t_album[x][f]
                              ) * pow(q_t_album[u][0] * pow(1 + q_t_album[u][0], -1, MOD), -1, MOD) % MOD
            q_tf[u][d] = x
            for i in range(n):
                if i == x:
                    continue
                if q_paper[u][d] & (1 << i):
                    q_tn[u][d].append(i)
                    q_t_album[u][d] = (q_t_album[u][d] + q_paper[u][d] * q_answer[i][0]) % MOD
            # print(u, u[n[0]], q_answer[u][n[1]], q_paper[u][n[1]], q_tn[u][n[1]], q_t_album[u][n[1]])
        # print("Done:", p, [u[n] for u in q_paper[p]])
        return q_answer[x][f]

    MOD_INV = pow((n - 1) * 2, -1, MOD)

    ans = [0] * n

    for i in range(n):
        ans[i] = dfs(i, 0, tree_from, q_answer, q_tn, q_t_album,
                     q_paper, q_tf)

    print(" ".join(map(str, ans)))


MOD = 998244353

main()