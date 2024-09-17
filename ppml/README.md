# PPML (Privacy Preserving Machine Learning)

C++ SGX/TDX attestation module for PPML

## Supported operating systems

* Ubuntu 20.04 LTS Server 64bits

## Requirements
 
1. [Install Intel SGX SDK](https://github.com/intel/linux-sgx#install-the-intelr-sgx-sdk)

2. Install SGX DCAP 1.16 libs

```bash
# Ubuntu 20.04, root
cd /opt/intel 
wget https://download.01.org/intel-sgx/sgx-dcap/1.16/linux/distro/ubuntu20.04-server/sgx_debian_local_repo.tgz 
tar xzf sgx_debian_local_repo.tgz 
echo 'deb [trusted=yes arch=amd64] file:///opt/intel/sgx_debian_local_repo focal main' | tee /etc/apt/sources.list.d/intel-sgx.list 
wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - 
apt-get update 

# Install DCAP Quote Verification lib for Attestation.sdkVerifyQuote()
apt-get -y install libsgx-dcap-quote-verify libsgx-dcap-quote-verify-dev
# Install DCAP TDX lib for Attestation.tdxGenerateQuote()
apt-get -y install libtdx-attest libtdx-attest-dev 

ln -s /usr/lib/x86_64-linux-gnu/libtdx_attest.so.1 /usr/lib/x86_64-linux-gnu/libtdx_attest.so
```

## Usage

```bash
export PPML_BUILD=true
mvn clean package
```

## (Optional) Rebuild after change JNI

Update JNI header

```bash
mvn clean package -DskipTests
javah -cp ppml-java-x86_64-linux/target/ppml-java-x86_64-linux-2.2.0-SNAPSHOT.jar com.intel.analytics.bigdl.ppml.dcap.Attestation
cp com_intel_analytics_bigdl_ppml_dcap_Attestation.h src/main/cpp
```

Check if shared lib is package into jar

```bash
mvn clean package
jar -tf ppml-java-x86_64-linux/target/ppml-java-x86_64-linux-2.2.0-SNAPSHOT.jar | grep libquote_verification.so
```

## Reference

* The `quote_verification.cpp` is tailored from [SGXDataCenterAttestationPrimitives/SampleCode/QuoteVerificationSample/App/App.cpp](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/master/SampleCode/QuoteVerificationSample/App)
