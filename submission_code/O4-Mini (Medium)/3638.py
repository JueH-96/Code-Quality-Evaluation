class Solution:
    def makeStringGood(self, s: str) -> int:
        # Compute frequency of each letter
        n = len(s)
        f = [0]*26
        for ch in s:
            f[ord(ch)-97] += 1

        INF = 10**18
        ans = INF

        # Try all segments T = [start .. start+k-1]
        for start in range(26):
            # max k so segment within 0..25
            for k in range(1, 26-start+1):
                # T is letters start ... start+k-1
                # collect candidate m values
                ms = set()
                # we require at least one of each, so m>=1
                ms.add(1)
                # add endpoints and midpoints
                for j in range(start, start+k):
                    # use f[j] and maybe f[j-1]
                    ms.add(f[j])
                    if j > 0:
                        ms.add(f[j-1])
                        # midpoint where u1 peaks
                        mid = (f[j] + f[j-1]) / 2.0
                        m0 = int(mid)
                        ms.add(m0)
                        ms.add(m0+1)
                # now filter m >=1
                m_cands = [m for m in ms if m >= 1]
                if not m_cands:
                    continue
                # We don't need m too large: max f[j] over T, beyond that cost grows linearly
                maxfj = 0
                for j in range(start, start+k):
                    if f[j] > maxfj:
                        maxfj = f[j]
                    if j>0 and f[j-1] > maxfj:
                        maxfj = f[j-1]
                # Restrict m up to maxfj+1 (one beyond to catch the boundary)
                m_cands = [m for m in m_cands if m <= maxfj+1]
                # Evaluate each candidate
                for m in m_cands:
                    # Compute U0 and U1
                    U0 = 0
                    U1 = 0
                    # We need to know u0 and u1 for each j to get rem
                    # Let's store u0 and u1 arrays
                    u0 = [0]*26
                    u1 = [0]*26
                    # First compute u0 for each j in T
                    for j in range(start, start+k):
                        u0[j] = m if f[j] >= m else f[j]
                        U0 += u0[j]
                    # Then compute u1 in increasing j
                    for j in range(start, start+k):
                        if j == 0:
                            # no predecessor
                            continue
                        # only if j in T and predecessor j-1 exists
                        if j < start+1:
                            continue
                        # rem from f[j-1]
                        rem = f[j-1] - u0[j-1] - u1[j-1]
                        if rem <= 0:
                            continue
                        # deficit at j
                        deficit = m - u0[j]
                        if deficit <= 0:
                            continue
                        # we can shift min(rem, deficit)
                        take = rem if rem < deficit else deficit
                        u1[j] = take
                        U1 += take
                    # compute cost = n + k*m - 2*U0 - U1
                    cost = n + k*m - 2*U0 - U1
                    if cost < ans:
                        ans = cost

        return ans