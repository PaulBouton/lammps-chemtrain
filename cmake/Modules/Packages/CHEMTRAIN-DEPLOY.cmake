set(CHEMTRAIN_DEPLOY_ROOT "" CACHE PATH "Path to chemtrain-deploy sources")
set(CHEMTRAIN_DEPLOY_LIBCONNECTOR "" CACHE FILEPATH "Path to the chemtrain-deploy connector library")

if(NOT CHEMTRAIN_DEPLOY_ROOT)
  message(FATAL_ERROR "PKG_CHEMTRAIN-DEPLOY requires CHEMTRAIN_DEPLOY_ROOT")
endif()
if(NOT CHEMTRAIN_DEPLOY_LIBCONNECTOR)
  message(FATAL_ERROR "PKG_CHEMTRAIN-DEPLOY requires CHEMTRAIN_DEPLOY_LIBCONNECTOR")
endif()
if(NOT EXISTS "${CHEMTRAIN_DEPLOY_ROOT}/connector/jcn_api.h")
  message(FATAL_ERROR "CHEMTRAIN_DEPLOY_ROOT must contain connector/jcn_api.h")
endif()
if(NOT EXISTS "${CHEMTRAIN_DEPLOY_ROOT}/third_party/dlpack/include/dlpack/dlpack.h")
  message(FATAL_ERROR "CHEMTRAIN_DEPLOY_ROOT must contain third_party/dlpack/include/dlpack/dlpack.h")
endif()
if(NOT EXISTS "${CHEMTRAIN_DEPLOY_LIBCONNECTOR}")
  message(FATAL_ERROR "CHEMTRAIN_DEPLOY_LIBCONNECTOR does not name an existing connector library")
endif()

add_library(LAMMPS::CHEMTRAIN_DEPLOY UNKNOWN IMPORTED)
set_target_properties(LAMMPS::CHEMTRAIN_DEPLOY PROPERTIES
  IMPORTED_LOCATION "${CHEMTRAIN_DEPLOY_LIBCONNECTOR}"
  INTERFACE_INCLUDE_DIRECTORIES "${CHEMTRAIN_DEPLOY_ROOT};${CHEMTRAIN_DEPLOY_ROOT}/third_party/dlpack/include")
target_link_libraries(lammps PRIVATE LAMMPS::CHEMTRAIN_DEPLOY)
