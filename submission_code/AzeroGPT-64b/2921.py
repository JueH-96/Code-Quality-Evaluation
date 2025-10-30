class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9+7

        def dp(x):
            x = int(x)
            dp_data = [1]*(len(str(x))+1)
            dp_data[0] = 0
            dp_data[1] = 10

            tmp = [1, 1]
            for s in range(2, len(str(x))+1):
                tmp.insert(0, tmp[0] + tmp[1])
                dp_data[s] = sum(tmp)

            lim = [0]*len(str(x))
            ref = [1]*len(str(x))

            for i in range(len(str(x))-1):
                if i == 0:
                    if str(x)[i] == "0":
                        continue
                lim[i] = (ref[i] * dp_data[len(str(x)) - i - 1]) % mod
                if i == 0:
                    ref[i] = int(str(x)[i]) * dp_data[len(str(x)) - i - 1] % mod
                    continue
                if int(str(x)[i]) - 1 >= 0 and int(str(x)[i]) - 1 <= 9:
                    ref[i] = (ref[i] * tmp[1] + ref[i-1] * tmp[0]) % mod
                else:
                    ref[i] = ref[i-1] * tmp[0]

                if i == len(str(x)) - 2:
                    if int(str(x)[i+1]) - int(str(x)[i]) in [1, -1] and int(str(x)[i+1]) >= 0 and int(str(x)[i+1]) <= 9:
                        lim[i] = (lim[i] + ref[i]*tmp[1] + ref[i+1]*tmp[0]) % mod
                    else:
                        lim[i] = (lim[i] + ref[i+1]*tmp[0]) % mod

            return lim[i]

        l_lim = dp(low)
        r_lim = dp(high)
        sen = len(str(high)) - len(str(low))
        if sen > 0:
            res = r_lim - l_lim
            for _ in range(sen):
                res = (res + dp(str(int(low) + 10**_))) % mod
        else:
            res = (r_lim - l_lim) % mod
            
        return (res + mod) % mod