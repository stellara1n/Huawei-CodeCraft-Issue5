//#define __infile__
//#define __debug__
//#define __arm__
#include <bits/stdc++.h>
#ifdef __arm__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif
#pragma GCC optimize(3)
using namespace std;
template <typename... Args>
void log(const char* format, Args... args) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    auto gmtime = std::gmtime(&tt);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", gmtime);
    printf("[%s.%03d] ", buffer, (int)(value.count() % 1000));
    int msg_size = snprintf(nullptr, 0, format, args...);
    char* msg = new char[msg_size + 1];
    snprintf(msg, msg_size + 1, format, args...);
    printf("%s", msg);
    delete[] msg;
}
#ifdef __debug__
    #define LOG(format, ...) log(format, ##__VA_ARGS__);
#else
    #define LOG(format, ...)
#endif

typedef long long ll;

const int MAXS = 1000000;
const int MAXN = 50000;
const int MAXD = 2000;
const int MAXQ = 800;
const int PRXD = 128;

char s[MAXS]; // 输入缓冲
int N, D;


float sort_d[MAXD][MAXN];
float sort_mid[MAXD];
uint64_t bit_d[MAXN][16];
float d[MAXN][MAXD]; // 原始数据
//float d_min,d_max,d_range;
float q[MAXQ][MAXD]; // 每次查询
uint64_t bit_q[MAXQ][16];
//uint8_t zd[MAXN][MAXD];
//uint8_t zq[MAXD];
int K;
/*
void parseFloats(char* str, float* a) {
    int cnt = 0;
    int len = (int)strlen(str);
    int pos = 0;
    while (pos < len) {
        if (isdigit(str[pos]) || str[pos] == '.' || str[pos] == '+' || str[pos] == '-') {
            bool negtive = false;
            ll integer_num = 0; // 整数部分
            float decimal_num = 0; // 小数部分
            float decimal_base = 0.1;
            bool scientific = false; // 是否带e
            ll scientific_num = 0; // e后面的数值
            while(pos < len) {
                if (str[pos] == '-') {
                    negtive = true;
                    pos++;
                    while(isdigit(str[pos])) {
                        integer_num = integer_num * 10 + str[pos] - '0';
                        pos++;
                    }
                } else if (str[pos] == '+'){
                    pos++;
                    while(isdigit(str[pos])) {
                        integer_num = integer_num * 10 + str[pos] - '0';
                        pos++;
                    }
                } else if (isdigit(str[pos])) {
                    while(isdigit(str[pos])) {
                        integer_num = integer_num * 10 + str[pos] - '0';
                        pos++;
                    }
                }
                if (str[pos] == '.') {
                    pos++;
                    while(isdigit(str[pos])) {
                        decimal_num = decimal_num + (str[pos] - '0') * decimal_base;
                        decimal_base *= 0.1;
                        pos++;
                    }
                } else if (str[pos] == 'e' || str[pos] == 'E') {
                    scientific = true;
                    pos++;
                    bool neg = false;
                    if(str[pos]=='-') {
                        neg = true;
                        pos++;
                    }
                    while(!isdigit(str[pos])) {
                        pos++;
                    }
                    while(isdigit(str[pos])) {
                        scientific_num = scientific_num * 10 + str[pos] - '0';
                        pos++;
                    }
                    if (neg) {
                        scientific_num *= -1;
                    }
                } else {
                    break;
                }
            }
            float result = integer_num + decimal_num;
            if(negtive) result *= -1;
            if (scientific) {
                if (scientific>0) {
                    result *= pow(10, scientific_num);
                } else {
                    result *= pow(0.1, -scientific_num);
                }
            }
            a[cnt++] = result;
        } else {
            pos++;
        }
    }
}
*/

//void compress_float_to_uint8(float* src, uint8_t* dst, int len, float min_value, float range) {
//    for (size_t i = 0; i< len; ++i) {
//        dst[i] = static_cast<uint8_t>(((src[i] - min_value) / range) * 64.0f);
//    }
//}

//int rseq[MAXD];
//float rd[MAXN][MAXD];
//float rq[MAXD];

__attribute__((target("avx2"))) float float_euclidean_distance(float* a, float* b, int len) {
#ifdef __arm__
    float square_sum = 0.0f;
    for (int i = 0; i < len; i += 4) {
        // 使用 vld1q_f32 加载浮点数子向量到 NEON 寄存器中
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_b = vld1q_f32(&b[i]);
        // 计算两个子向量的差
        float32x4_t diff = vsubq_f32(vec_a, vec_b);
        // 计算差的平方
        float32x4_t square_diff = vmulq_f32(diff, diff);
        // 将平方差的各分量相加
        float32x2_t sum_low_high = vadd_f32(vget_low_f32(square_diff), vget_high_f32(square_diff));
        float32x2_t pairwise_sum = vpadd_f32(sum_low_high, sum_low_high);
        float sum = vget_lane_f32(pairwise_sum, 0);
        // 累加子向量的平方差之和
        square_sum += sum;
    }
    // 计算平方根得到欧氏距离
    return square_sum;
#else
    float distance_squared = 0;
    // 使用 AVX2 指令集加速计算
    for (int i = 0; i < len; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(a_vec, b_vec);
        __m256 diff_squared = _mm256_mul_ps(diff, diff);
        __m256 sum = _mm256_hadd_ps(diff_squared, diff_squared);
        __m128 sum_lo = _mm256_castps256_ps128(sum);
        __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
        __m128 sum_128 = _mm_add_ps(sum_lo, sum_hi);
        float distance_squared_partial = _mm_cvtss_f32(_mm_hadd_ps(sum_128, sum_128));
        distance_squared += distance_squared_partial;
    }
    return distance_squared;
#endif
}


//__attribute__((target("avx2")))  uint32_t uint8_euclidean_distance(const uint8_t* a, const uint8_t* b, int len) {
//#ifdef __arm__
//    uint32_t distance_squared = 0;
//    // 使用 NEON 指令集加速计算
//    for (int i = 0; i < len; i += 16) {
//        uint8x16_t a_vec = vld1q_u8(&a[i]);
//        uint8x16_t b_vec = vld1q_u8(&b[i]);
//        int16x8_t diff_low = vsubl_s8(vget_low_s8(vreinterpretq_s8_u8(a_vec)), vget_low_s8(vreinterpretq_s8_u8(b_vec)));
//        int16x8_t diff_high = vsubl_s8(vget_high_s8(vreinterpretq_s8_u8(a_vec)), vget_high_s8(vreinterpretq_s8_u8(b_vec)));
//        int16x8_t diff_squared_low = vmulq_s16(diff_low, diff_low);
//        int16x8_t diff_squared_high = vmulq_s16(diff_high, diff_high);
//        uint32x4_t diff_squared_low_u32 = vreinterpretq_u32_s32(vaddl_s16(vget_low_s16(diff_squared_low), vget_high_s16(diff_squared_low)));
//        uint32x4_t diff_squared_high_u32 = vreinterpretq_u32_s32(vaddl_s16(vget_low_s16(diff_squared_high), vget_high_s16(diff_squared_high)));
//        uint32_t distance_squared_low = vaddvq_u32(diff_squared_low_u32);
//        uint32_t distance_squared_high = vaddvq_u32(diff_squared_high_u32);
//        distance_squared += distance_squared_low + distance_squared_high;
//    }
//    return distance_squared;
//#else
//    uint32_t distance_squared = 0;
//    // 使用 AVX2 指令集加速计算
//    for (int i = 0; i < len; i += 32) {
//        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
//        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
//        __m256i diff = _mm256_sub_epi8(a_vec, b_vec);
//        __m256i diff_squared = _mm256_maddubs_epi16(diff, diff);
//        __m256i diff_squared_lo = _mm256_unpacklo_epi16(diff_squared, _mm256_setzero_si256());
//        __m256i diff_squared_hi = _mm256_unpackhi_epi16(diff_squared, _mm256_setzero_si256());
//        __m256i sum = _mm256_add_epi32(diff_squared_lo, diff_squared_hi);
//        __m128i sum_lo = _mm256_castsi256_si128(sum);
//        __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
//        __m128i sum_128 = _mm_add_epi32(sum_lo, sum_hi);
//        uint32_t distance_squared_partial = _mm_cvtsi128_si32(_mm_hadd_epi32(sum_128, sum_128));
//        distance_squared += distance_squared_partial;
//    }
//    return distance_squared;
//#endif
//}


void prepare() {
    // 方差排序
//    std::vector<float> columnMeans(N, 0.0);
//    std::vector<pair<float, int> > columnVariances(D, pair<float, int>(0.0, 0));
//    for (int j = 0; j < D; j++) {
//        for (int i = 0; i < N; i++) {
//            columnMeans[j] += d[i][j];
//        }
//        columnMeans[j] /= N;
//    }
//    for (int j = 0; j < D; j++) {
//        for (int i = 0; i < N; i++) {
//            columnVariances[j].first += pow(d[i][j] - columnMeans[j], 2);
//        }
//        columnVariances[j].first /= N;
//        columnVariances[j].first *= -1.0;
//        columnVariances[j].second = j;
//    }
//    sort(columnVariances.begin(), columnVariances.end());
//    for(int j=0;j<D;j++) {
//        rseq[columnVariances[j].second] = j;
//    }
//    for(int i=0;i<N;i++) {
//        for(int j=0;j<D;j++) {
//            rd[i][rseq[j]] = d[i][j];
//        }
//    }
    // 压缩数据
//    d_min = d[0][0];
//    d_max = d[0][0];
//    for(int i=0;i<N;i++) {
//        for(int j=0;j<min(D,PRXD);j++) {
//            if (rd[i][j] > d_max) {
//                d_max = rd[i][j];
//            }
//            if (rd[i][j] < d_min) {
//                d_min = rd[i][j];
//            }
//        }
//    }
//    d_range = d_max - d_min;
//    for(int i=0;i<N;i++) {
//        compress_float_to_uint8(d[i], zd[i], min(D,PRXD), d_min, d_range);
//    }
    
    for(int i=0;i<N;i++) {
        for(int j=0;j<D;j++) {
            sort_d[j][i] = d[i][j];
        }
    }
    for(int j=0;j<D;j++) {
        sort(sort_d[j], sort_d[j]+N);
        sort_mid[j] = sort_d[j][N/2];
    }
    
    for(int i=0;i<N;i++) {
        for(int j=0;j<D;j++) {
            if(d[i][j]>sort_mid[j]) {
                bit_d[i][j/64] |= (1ULL<<(j%64));
            }
        }
    }
    
    
    
    
    
    printf("ok\n");
    fflush(stdout);
    
}

template<typename T>
struct TopK {
    int _k;
    priority_queue<T, std::vector<T>, std::less<T > > _pq;
    void set(int k) {
        _k = k;
    }
    void add(T value) {
        if (_pq.size() < _k) {
            _pq.push(value);
        } else if (value < _pq.top()) {
            _pq.pop();
            _pq.push(value);
        }
    }
    bool empty() {
        return _pq.size() == 0;
    }
    T top() {
        return _pq.top();
    }
    void pop() {
        _pq.pop();
    }
};
TopK<pair<float,int> > top1K;
//TopK<pair<uint32_t, int> > top100K;
TopK<pair<int, int> > top1000K;

inline int hamming_distance(const uint64_t* a, const uint64_t* b) {
    int dis = 0;
    for(int i=0;i<16;i++) {
        dis += __builtin_popcountll(a[i] xor b[i]);
    }
    return dis;
}

void solve(int qid) {
//    for(int j=0;j<D;j++) {
//        rq[rseq[j]] = q[j];
//    }
//    compress_float_to_uint8(rq, zq, min(D,PRXD), d_min, d_range);
//    for(int i=0;i<N;i++) {
//        uint32_t dis = uint8_euclidean_distance(zd[i], zq, min(D,PRXD));
//        top100K.add(pair<uint32_t, int>(dis, i));
//    }
    
//    while(!top100K.empty()) {
//        int i = top100K.top().second;
//        top100K.pop();
//        float dis = float_euclidean_distance(rd[i], rq, D);
//        top1K.add(pair<float, int>(dis, i));
//    }
    
    for(int j=0;j<D;j++) {
        if(q[qid][j]>sort_mid[j]) {
            bit_q[qid][j/64] |= (1ULL<<(j%64));
        }
    }
    
    for(int i=0;i<N;i++) {
        int dis = hamming_distance(bit_d[i], bit_q[qid]);
        top1000K.add(pair<int, int>(dis, i));
    }
    while(!top1000K.empty()) {
        int i = top1000K.top().second;
        float dis = float_euclidean_distance(d[i], q[qid], D);
        top1000K.pop();
        top1K.add(pair<float, int>(dis, i));
    }
    bool first = true;
    while(!top1K.empty()) {
        if (!first) printf(" ");
        printf("%d", top1K.top().second);
        first = false;
        top1K.pop();
    }
    printf("\n");
    fflush(stdout);
}

int main() {
#ifdef __infile__
    freopen("in.txt", "r", stdin);
#endif
    scanf("%d%d", &N, &D);
    LOG("%d %d\n", N, D)
    for(int i=0;i<N;i++) {
        // fgets(s, MAXS, stdin);
        // parseFloats(s, d[i]);
        for(int j=0;j<D;j++) {
            scanf("%f", &d[i][j]);
        }
    }
    LOG("read end\n")
    prepare();
    LOG("prepare end\n")
    scanf("%d\n", &K);
    top1K.set(K);
    //top100K.set(100*K);
    top1000K.set(100*K);
    int qid = 0;
    bool end = false;
    while(true) {
        //fgets(s, MAXS, stdin);
//        if(s[0]=='e') {
//            break;
//        }
        //parseFloats(s, q[qid]);
        for(int j=0;j<D;j++) {
            if(scanf("%f", &q[qid][j])==0) {
                end = true;
            }
        }
        if (end) break;
        solve(qid++);
    }
    LOG("query end\n")
    return 0;
}
