#include<bits/stdc++.h>
using namespace std;

vector<int> vt[1005];
int main()
{
    /*
    totalTuningTimes equals the total tuning times.
    tuningOperatorNums equals the number of tuned operators.
    */
    int totalTuningTimes,tuningOperatorNums;
    scanf("%d%d",&totalTuningTimes,&tuningOperatorNums);
    for(int i = 1;i<=totalTuningTimes;++i)
    {
        for(int j = 1;j<=tuningOperatorNums;++j)
        {
            // tmp euqals the optimal level of parallelism, given by DS2
            int tmp;
            scanf("%d",&tmp);
            vt[i-1].push_back(tmp);
        }
    }
    int ans = 0;

    srand(time(0));
    for(int i = 1;i<=totalTuningTimes;++i)
    {
        int ansNow = 0;
        for(int j = 1;j<=tuningOperatorNums;++j)
        {
            int cntNow = 0;
            set<int> st;
            while(1)
            {
                int maximalBound = 32;
                int tmp = (rand()%maximalBound)+1;
                if(st.count(tmp)) continue;
                cntNow++;
                st.insert(tmp);
                if(tmp==vt[i-1][j-1])
                {
                    break;
                }
            }
            // For many operators, because we concurrently tune all operators, so we sum the maximal reconfigurations at each tuning time.
            ansNow = max(ansNow,cntNow);
        }
        ans += ansNow;
    }

    printf("%d\n",ans);
    return 0;
}
