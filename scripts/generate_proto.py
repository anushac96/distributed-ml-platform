import subprocess
import os

def generate_grpc_code():
    """Generate Python gRPC code from .proto files"""
    proto_dir = "proto"
    output_dir = "src/generated"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each proto file
    proto_files = ["master.proto", "worker.proto", "parameter_server.proto"]
    
    for proto_file in proto_files:
        if os.path.exists(os.path.join(proto_dir, proto_file)):
            cmd = [
                "python", "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                proto_file
            ]
            
            print(f"Generating code for {proto_file}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Generated code for {proto_file}")
            else:
                print(f"✗ Error generating {proto_file}:")
                print(result.stderr)

if __name__ == "__main__":
    generate_grpc_code()