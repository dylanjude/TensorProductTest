void cutensorWrap(occa::device &device, int M, int N, int K,
                  const std::vector<double> &Ar, const std::vector<double> &As,
                  const std::vector<double> &At, const std::vector<double> &B,
                  std::vector<double> &C) {
#ifdef DGX3D_HAS_CUTENSOR
  occa::memory d_Ar = device.malloc<double>(Ar.size(), Ar.data());
  occa::memory d_As = device.malloc<double>(As.size(), As.data());
  occa::memory d_At = device.malloc<double>(At.size(), At.data());
  occa::memory d_B = device.malloc<double>(B.size(), B.data());
  occa::memory d_C = device.malloc<double>(C.size(), C.data());
  occa::memory d_tmpJ = device.malloc<double>(M * M * K * N);

  double *Arptr = static_cast<double *>(d_Ar.ptr());
  double *Asptr = static_cast<double *>(d_As.ptr());
  double *Atptr = static_cast<double *>(d_At.ptr());
  double *Bptr = static_cast<double *>(d_B.ptr());
  double *tptr = static_cast<double *>(d_tmpJ.ptr());
  double *Cptr = static_cast<double *>(d_C.ptr());

  auto cutp = cutensorProduct(M, K, N);

#if TIMER
  Timer stopwatch;
  for (int itry = 0; itry < NTRYS; itry++) {
  if (itry ==1) stopwatch.tick();
#endif
    cutp.contract(Arptr, Asptr, Atptr, Bptr, tptr, Cptr);
    // tensorProduct_cuTENSOR(M,K,N,
    // Arptr,Asptr,Atptr,
    // Bptr,tptr,Cptr);
#if TIMER
  }
  auto duration=stopwatch.tock()/ (NTRYS-1);
  double FLOPS = (2*N-1)*(K*K*M + K*M*M + M*M*M)/duration/1e12;
  printf("cutensor time = %e TFLOPS=%.2f\n", duration,FLOPS);
#endif
  d_C.copyTo(C.data());
#endif
}
