import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u,v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    INF_NEG = -10**18
    # We'll store for each node two dicts:
    # dp0[node] = {1: (no,yes), 4:(no,yes)}  for parent_selected=0
    # dp1[node] = {0: (no,yes), 3:(no,yes)}  for parent_selected=1
    dp0_no = [{} for _ in range(n+1)]
    dp0_yes = [{} for _ in range(n+1)]
    dp1_no = [{} for _ in range(n+1)]
    dp1_yes = [{} for _ in range(n+1)]
    seen = [False]*(n+1)
    def dfs(u):
        seen[u]=True
        # accumulate child gains
        gains_no = []
        gains_yes = []
        for v in adj[u]:
            if not seen[v]:
                dfs(v)
                # child's dp1 possibilities are at k=0 or k=3
                # dp1[0]: child has k=0 => total deg=1 => not internal
                # dp1[3]: child has k=3 => total deg=4 => internal
                # get their values
                no0 = dp1_no[v].get(0, INF_NEG)
                yes0 = dp1_yes[v].get(0, INF_NEG)
                no3 = dp1_no[v].get(3, INF_NEG)
                yes3 = dp1_yes[v].get(3, INF_NEG)
                # combine: for no_gain we take max of those that give no internal
                # that's only dp1_no[0]
                best_no = no0
                # for yes_gain, could be child's own internal: dp1_yes[3]
                best_yes = yes3
                # It's possible using dp1_no[3]? That has yes0? but dp1_no[3] is invalid since k=3 gives internal
                # So we stick with this
                gains_no.append(best_no)
                gains_yes.append(best_yes)
        m = len(gains_no)
        # sort no_gains descending, keep original indexes for yes
        idxs = list(range(m))
        idxs.sort(key=lambda i: gains_no[i], reverse=True)
        # prepare prefix sums of top up to 4
        prefix = [0]
        for t in range(min(4,m)):
            prefix.append(prefix[-1] + gains_no[idxs[t]])
        # dp0: k=1 and k=4
        # k=1
        # no: pick top1 no_gain
        if m >= 1:
            no1 = prefix[1] + 1
        else:
            no1 = INF_NEG
        # yes: current cannot be internal (k!=4), so need child internal: pick any gains_yes[i]
        best_child_yes = INF_NEG
        for i in range(m):
            if gains_yes[i] > INF_NEG:
                # if we pick that child and k=1, no other child
                best_child_yes = max(best_child_yes, gains_yes[i] + 1)
        yes1 = best_child_yes
        dp0_no[u][1] = no1
        dp0_yes[u][1] = yes1
        # k=4
        if m >= 4:
            no4 = prefix[4] + 1
        else:
            no4 = INF_NEG
        # yes: either current is internal => sum top4 no +1
        if no4 > INF_NEG:
            yes4 = no4
        else:
            yes4 = INF_NEG
        # or one child internal: for each child i with yes, take yes + sum top3 no from others
        if m >= 4:
            # we need sum of top3 excluding that child
            # Precompute for each idx the best 3 from others:
            # But m can be big, but k small: we can do per child in O(4)
            top4_idx = idxs[:4]
            sum4 = prefix[4]
            for i in range(m):
                gy = gains_yes[i]
                if gy <= INF_NEG: continue
                # is i among top4?
                if i in top4_idx:
                    # sum3 = sum of top4 no minus gains_no[i]
                    sum3 = sum4 - gains_no[i]
                else:
                    # sum3 = sum of top3 no
                    sum3 = prefix[3]
                yes4 = max(yes4, gy + sum3 + 1)
        dp0_no[u][4] = no4
        dp0_yes[u][4] = yes4
        # dp1: k=0 and k=3
        # k=0
        dp1_no[u][0] = 1  # leaf, no internal
        dp1_yes[u][0] = INF_NEG
        # k=3
        if m >= 3:
            sum3 = prefix[3]
            # total degree =3+1=4 => internal
            dp1_no[u][3] = INF_NEG
            dp1_yes[u][3] = sum3 + 1
        else:
            dp1_no[u][3] = INF_NEG
            dp1_yes[u][3] = INF_NEG

    dfs(1)
    # answer is max over u of dp0_yes[u][1] and dp0_yes[u][4]
    ans = INF_NEG
    for u in range(1,n+1):
        a1 = dp0_yes[u].get(1, INF_NEG)
        a4 = dp0_yes[u].get(4, INF_NEG)
        if a1 > ans: ans = a1
        if a4 > ans: ans = a4
    if ans < 0:
        print(-1)
    else:
        print(ans)

if __name__=='__main__':
    main()