class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        N, K = len(s), k 

        numEditsNeeded = collections.defaultdict(int)
        for length in range(1, N):
            sToCheck = s[:length]
            start, end = 0, len(sToCheck)-1
            while start < end:
                if sToCheck[start] != sToCheck[end]:
                    numEditsNeeded[sToCheck] += 1
                start += 1
                end -= 1
            for i in range(1, length):
                sToCheck = s[i:(i+length)]
                if len(sToCheck) == 1:
                    continue
                start, end = 0, len(sToCheck)-1
                while start < end:
                    if sToCheck[start] != sToCheck[end]:
                        numEditsNeeded[sToCheck] += 1    
                    start += 1
                    end -= 1

        #print('numEditsNeeded', numEditsNeeded)

        minChangesPerLen = []
        for i in range(2*N-1):
            minChangesPerLen.append(float("inf"))
        for sToCheck in numEditsNeeded:
            minChangesPerLen[len(sToCheck)-1] = min(minChangesPerLen[len(sToCheck)-1], numEditsNeeded[sToCheck])
            if len(sToCheck) > 1 and len(sToCheck) % 2 == 0 and numEditsNeeded[sToCheck] == 0:
                minChangesPerLen[len(sToCheck)//2-1] = 0

        minChanges, idx = 0, 0
        while idx <= N-K:
            minChangesForK = float("inf")
            for lastSubstringLenIdx in range(min(K-1, N-idx-1), N-idx):
                minChangesForK = min(minChangesForK, minChangesPerLen[lastSubstringLenIdx] + minChangesPerLen[lastSubstringLenIdx+idx])
            minChanges += minChangesForK 
            idx += 1
        return minChanges if minChanges != float("inf") else N