# -*- CMake -*- file for testing the direct calls to individual C++ classes

add_executable(test_lammps_class test_lammps_class.cpp)
target_link_libraries(test_lammps_class PRIVATE lammps GTest::GMock)
add_test(NAME LammpsClass COMMAND test_lammps_class)
set_tests_properties(LammpsClass PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable(test_input_class test_input_class.cpp)
target_link_libraries(test_input_class PRIVATE lammps GTest::GTestMain)
add_test(NAME InputClass COMMAND test_input_class)

add_executable(test_advanced_utils test_advanced_utils.cpp)
target_link_libraries(test_advanced_utils PRIVATE lammps GTest::GMock)
add_test(NAME AdvancedUtils COMMAND test_advanced_utils)

add_executable(test_error_class test_error_class.cpp)
target_link_libraries(test_error_class PRIVATE lammps GTest::GMock)
add_test(NAME ErrorClass COMMAND test_error_class)

add_executable(test_kspace_styles test_kspace_styles.cpp)
target_link_libraries(test_kspace_styles PRIVATE lammps GTest::GMock)
add_test(NAME KSpaceStyles COMMAND test_kspace_styles)
set_tests_properties(KSpaceStyles PROPERTIES LABELS "slow"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable(test_neighbor_class test_neighbor_class.cpp)
target_link_libraries(test_neighbor_class PRIVATE lammps GTest::GMock)
if(PKG_OPENMP)
  target_compile_definitions(test_neighbor_class PRIVATE -DLMP_OPENMP)
endif()
add_test(NAME NeighborClass COMMAND test_neighbor_class)
set_tests_properties(NeighborClass PROPERTIES ENVIRONMENT "LAMMPS_POTENTIALS=${LAMMPS_POTENTIALS_DIR}")

if(PKG_CHEMTRAIN-DEPLOY)
  add_executable(test_chemtrain_settings test_chemtrain_settings.cpp)
  target_include_directories(test_chemtrain_settings PRIVATE
    ${LAMMPS_SOURCE_DIR}/CHEMTRAIN-DEPLOY)
  target_link_libraries(test_chemtrain_settings PRIVATE lammps GTest::GMock
    ${CHEMTRAIN_DEPLOY_LIBCONNECTOR})
  add_test(NAME ChemtrainSettings COMMAND test_chemtrain_settings)
  add_mpi_test(NAME ChemtrainSettingsMPI NUM_PROCS 2
               COMMAND $<TARGET_FILE:test_chemtrain_settings>
                       --gtest_filter=ChemtrainSettingsTest.ReadsBinaryModelCollectively)

  add_executable(test_chemtrain_output test_chemtrain_output.cpp)
  target_include_directories(test_chemtrain_output PRIVATE
    ${LAMMPS_SOURCE_DIR}/CHEMTRAIN-DEPLOY
    ${LAMMPS_SOURCE_DIR}/KOKKOS
    ${CHEMTRAIN_DEPLOY_ROOT}
    ${CHEMTRAIN_DEPLOY_ROOT}/third_party/dlpack/include)
  target_link_libraries(test_chemtrain_output PRIVATE lammps GTest::GMock
    ${CHEMTRAIN_DEPLOY_LIBCONNECTOR})
  if(PKG_KOKKOS)
    target_compile_definitions(test_chemtrain_output PRIVATE
      LMP_TEST_KOKKOS_OUTPUT)
    target_link_libraries(test_chemtrain_output PRIVATE Kokkos::kokkos)
  endif()
  add_test(NAME ChemtrainOutput COMMAND test_chemtrain_output)
  string(CONCAT CHEMTRAIN_OUTPUT_MPI_FILTER
    "--gtest_filter=ChemtrainOutputTest.LocalScalarIsSummedAcrossRanks"
    ":ChemtrainOutputTest.GlobalVectorIsNotReduced")
  add_mpi_test(NAME ChemtrainOutputMPI NUM_PROCS 2
               COMMAND $<TARGET_FILE:test_chemtrain_output>
                       ${CHEMTRAIN_OUTPUT_MPI_FILTER})

  if(PKG_MOLECULE)
    add_executable(test_chemtrain_topology test_chemtrain_topology.cpp)
    target_include_directories(test_chemtrain_topology PRIVATE
      ${LAMMPS_SOURCE_DIR}/CHEMTRAIN-DEPLOY
      ${LAMMPS_SOURCE_DIR}/KOKKOS
      ${CHEMTRAIN_DEPLOY_ROOT}
      ${CHEMTRAIN_DEPLOY_ROOT}/third_party/dlpack/include)
    target_link_libraries(test_chemtrain_topology PRIVATE lammps GTest::GMock)
    if(PKG_KOKKOS AND Kokkos_ENABLE_CUDA)
      target_compile_definitions(test_chemtrain_topology PRIVATE LMP_TEST_KOKKOS_HOST)
      target_link_libraries(test_chemtrain_topology PRIVATE Kokkos::kokkos)
    endif()
    add_test(NAME ChemtrainTopology COMMAND test_chemtrain_topology)
    string(CONCAT CHEMTRAIN_TOPOLOGY_MPI_FILTER
      "--gtest_filter=-LAMMPSTest.ChemtrainParticleInputUsesScalarIntegerCustomProperties"
      ":NewtonBond/ChemtrainTopologyTest.PreservesShortestCategoryAndRejectsPeriodicImage/*")
    add_mpi_test(NAME ChemtrainTopologyMPI NUM_PROCS 2
                 COMMAND $<TARGET_FILE:test_chemtrain_topology>
                         ${CHEMTRAIN_TOPOLOGY_MPI_FILTER})
  endif()
endif()
