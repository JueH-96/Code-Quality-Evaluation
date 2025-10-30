class Solution:
    def findLatestTime(self, s: str) -> str:
        s_list = list(s)
        h0, h1 = s_list[0], s_list[1]
        m0, m1 = s_list[3], s_list[4]

        # Process hour digits
        if h0 == '?':
            if h1 == '?':
                h0, h1 = '1', '1'
            else:
                if h1 <= '1':
                    h0 = '1'
                else:
                    h0 = '0'
        else:
            if h1 == '?':
                if h0 == '1':
                    h1 = '1'
                else:
                    h1 = '9'

        # Process minute digits
        if m0 == '?':
            m0 = '5'
        if m1 == '?':
            m1 = '9'

        s_list[0], s_list[1] = h0, h1
        s_list[3], s_list[4] = m0, m1

        return ''.join(s_list)