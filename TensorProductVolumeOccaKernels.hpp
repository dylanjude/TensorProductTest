int nearestPower2(int n) {
    if ((n & (n - 1)) == 0) { // Correct power of 2 check
      return n;
    }
    // Set all bits to the right of the most significant bit to 1
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
};

void TensorProductVolumeOCCA(occa::device &device, int M, int MPad, int N,
			     int K, int LDB, int LDC,
			     const std::vector<double> &Ar,
			     const std::vector<double> &As,
			     const std::vector<double> &At,
			     const std::vector<double> &B,
			     std::vector<double> &C, bool addToC) {

  occa::memory d_Ar = device.malloc<double>(Ar.size(), Ar.data());
  occa::memory d_As = device.malloc<double>(As.size(), As.data());
  occa::memory d_At = device.malloc<double>(At.size(), At.data());
  occa::memory d_B = device.malloc<double>(B.size(), B.data());
  occa::memory d_C = device.malloc<double>(C.size(), C.data());
  occa::memory d_tmpK = device.malloc<double>(K * K * MPad * N);
  occa::memory d_tmpJ = device.malloc<double>(M * K * MPad * N);

  std::string kpath4 = 
      std::string(DGX3D_OKL_DIR) + "/TensorProductAll.okl";
  std::string kpath5 = 
      std::string(DGX3D_OKL_DIR) + "/TensorProductAllGeneric.okl";
  std::string kpath6 =
    std::string(DGX3D_OKL_DIR) + "/TensorProductVolumeFused_Padded.okl";
  // Build kernel with appropriate properties
  occa::properties kernelProps;
  // Define the properties for this kernel.
  occa::properties props,props2;
  //[4,4,4,65536] -> [8,8,8,65536]

  // build the power 2 kernel
  // TODO try including TPB in to this
  //bool pow2kernel= ((M & (M-1)) == 0 && (K & (K-1)) == 0 && M==2*K);
  bool pow2kernel= (M==2*K) || (K==2*M);
  props["defines/TPB"] = K * K * MPad;
  props["defines/M"] = MPad;
  props["defines/K"] = K;
  props["defines/N"] = N;
  props["compiler_flags"] = "-O3"; // -lineinfo";
  occa::kernel kPow2;
  if (M > K) {
    kPow2 = device.buildKernel(kpath4,"TensorProductVolumeAll_M_gt_K",props);
  }
  else {
    kPow2 = device.buildKernel(kpath4,"TensorProductVolumeAll_K_gt_M",props);
  }
  
  // build the generic kernel
  occa::kernel kGeneric;
  int TPB = 256;
  int Mp = nearestPower2(M);
  int Kp = nearestPower2(K);
  /*
  if (M > K) {
    Kp = K;
    Mp = 2*K;
  }
  else {
    Mp = M;
    Kp =2*M;
  }*/
  props2["compiler_flags"] = "-O3"; // -lineinfo";
  props2["defines/MPad"] = Mp;
  props2["defines/KPad"] = Kp;
  if (M > K) {
    int Np = N*(Mp*Kp*Kp)/TPB;
    while (Np > N) {
      TPB*=2;
      Np = N*(Mp*Kp*Kp)/TPB;
    }
    props2["defines/blockDimX"] = TPB/(Mp*Kp);
    if (!pow2kernel) printf("Using generic kernel with (Mp,Kp,Np)=(%d,%d,%d) for (M,K,N)=(%d %d %d)\n",Mp,Kp,Np,M,K,N);
    if (!pow2kernel) printf("Using TPB = %d and fac=%d\n",TPB,static_cast<int>(std::round(N/Np)));
    props2["defines/Nblocks"] = Np;
    props2["defines/fac"] = static_cast<int>(std::round(N/Np));
    kGeneric = device.buildKernel(kpath5, "TensorProductVolumeAllGeneric_M_gt_K", props2);
  } else {
    int Np = N*(Mp*Mp*Mp)/TPB;
    while (Np > N) {
      TPB*=2;
      Np = N*(Mp*Mp*Mp)/TPB;
    }
    props2["defines/blockDimX"] = TPB/(Mp*Mp);
    if (!pow2kernel) printf("Using generic kernel with (Mp,Kp,Np)=(%d,%d,%d) for (M,K,N)=(%d %d %d)\n",Mp,Kp,Np,M,K,N);
    if (!pow2kernel) printf("Using TPB = %d and fac=%d\n",TPB,static_cast<int>(std::round(N/Np)));
    props2["defines/Nblocks"] = Np;
    props2["defines/fac"] = static_cast<int>(std::round(N/Np));
    kGeneric = device.buildKernel(kpath5, "TensorProductVolumeAllGeneric_K_gt_M", props2);
  }

  occa::kernel kTest;
  occa::properties props3;
  props3["compiler_flags"] = "-O3"; // -lineinfo";
  props3["defines/MPad"] = nearestPower2(M);
  props3["defines/KPad"] = nearestPower2(K);
  props3["defines/TPad"] = std::max(nearestPower2(M),nearestPower2(K));
  props3["defines/fac"] = 1;
  printf("MPad,KPad,TPad=%d %d %d\n",nearestPower2(M),nearestPower2(K),std::max(nearestPower2(M),nearestPower2(K)));
  kTest = device.buildKernel(kpath6, "TensorProductVolumeFusedFinal", props3);
  
#if TIMER
  Timer stopwatch;
  stopwatch.tick();
  int add_to_C = addToC;
  for (int itry = 0; itry < NTRYS; itry++) {
#endif
    //kTest(M,N,K,LDB,LDC,d_Ar,d_As,d_At,d_B,d_C,0,0,0,add_to_C);
#if 1
    if (pow2kernel) {
      kPow2(LDB,LDC,d_Ar,d_As,d_At,d_B,d_C,0,0,0,add_to_C);
    }
    else {
      kGeneric(M,N,K,LDB,LDC,d_Ar,d_As,d_At,d_B,d_C,0,0,0,add_to_C);
    }
#endif
#if TIMER
  }
  auto duration=stopwatch.tock()/ NTRYS;
  double FLOP = (double)N * 2.0 * K * (K*K*M + K*M*M + M*M*M);
  double FLOPS = FLOP/duration/1e12;
  double BYTES = N*(8*K*K*K + 3*K*M*8 + 8*M*M*M);
  printf("Arithmetic intensity (FLOP/BYTES) = %f\n", FLOP/BYTES);
  printf("OCCA unified kernel Compute time = %e TFLOPS=%.2f\n", duration, FLOPS);
#endif
  // copy back
  d_C.copyTo(C.data());
}

void TensorProductVolumeSplit_OCCA(occa::device &device, int M, int MPad, int N,
                                   int K, int LDB, int LDC,
                                   const std::vector<double> &Ar,
                                   const std::vector<double> &As,
                                   const std::vector<double> &At,
                                   const std::vector<double> &B,
                                   std::vector<double> &C, bool addToC) {

  occa::memory d_Ar = device.malloc<double>(Ar.size(), Ar.data());
  occa::memory d_As = device.malloc<double>(As.size(), As.data());
  occa::memory d_At = device.malloc<double>(At.size(), At.data());
  occa::memory d_B = device.malloc<double>(B.size(), B.data());
  occa::memory d_C = device.malloc<double>(C.size(), C.data());
  occa::memory d_tmpK = device.malloc<double>(K * K * MPad * N);
  occa::memory d_tmpJ = device.malloc<double>(M * K * MPad * N);

  std::string kpath1 =
      std::string(DGX3D_OKL_DIR) + "/TensorProductVolumeStage1.okl";
  std::string kpath2 =
      std::string(DGX3D_OKL_DIR) + "/TensorProductVolumeStage2.okl";
  std::string kpath3 =
      std::string(DGX3D_OKL_DIR) + "/TensorProductVolumeStage3.okl";
  // Build kernel with appropriate properties
  occa::properties kernelProps;
  // Define the properties for this kernel.
  occa::properties props;
  //[4,4,4,16384] -> [8,8,8,16384]

  props["defines/TPB"] = K * K * MPad;
  props["defines/MPad"] = MPad;
  props["compiler_flags"] = "-O3";// -lineinfo";
  occa::settings()["verbose"] = true;
  occa::kernel kInit = device.buildKernel(kpath1, "InitCToZero", props);
  occa::kernel kStage1 =
      device.buildKernel(kpath1, "TensorProductVolumeStage1", props);
  props["defines/TPB"] = K * M * MPad;
  occa::kernel kStage2 =
      device.buildKernel(kpath2, "TensorProductVolumeStage2", props);
  props["defines/TPB"] = M * M * MPad;
  occa::kernel kStage3 =
      device.buildKernel(kpath3, "TensorProductVolumeStage3", props);
#if TIMER
  Timer stopwatch;
  stopwatch.tick();
  int add_to_C = addToC;
  for (int itry = 0; itry < NTRYS; itry++) {
#endif
    if (addToC)
      kInit(M, N, LDC, 0, d_C);
    // Launch tiled kernels
    kStage1(M, N, K, LDB, d_At, d_B, d_tmpK, N * K * K * MPad, 0, 0);
    kStage2(M, N, K, d_As, d_tmpK, d_tmpJ, N * K * MPad * MPad, 0);
    kStage3(M, N, K, d_Ar, d_tmpJ, d_C, LDC, N * MPad * MPad * MPad, 0, 0,
          add_to_C);
#if TIMER
  }
  printf("OCCA Split kernels Compute time = %e\n", stopwatch.tock() / NTRYS);
#endif
  // copy back
  d_C.copyTo(C.data());
}
