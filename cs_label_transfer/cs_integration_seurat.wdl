version 1.0
workflow cs_integration_seurat {
    input {
        File ref_obj
        File query_obj
        String ref_name
        String query_name
    }
    call process_anndata {
        input:
            ref = ref_obj,
            query = query_obj,
            ref_name = ref_name,
            query_name = query_name
    }

    call integration {
        input:
            ref_dir = process_anndata.ref_dir,
            query_dir = process_anndata.query_dir,
            ref_name = ref_name,
            query_name = query_name
    }
}

task process_anndata {
    input { 
        File ref
        File query
        String ref_name
        String query_name
    }
    command <<<
    	wget https://raw.githubusercontent.com/mhl34/macosko_scripts/master/create_seurat_obj.py
    
        python create_seurat_obj.py --ref ~{ref} --query ~{query} --ref_name ~{ref_name} --query_name ~{query_name}

        ref_dir=`dirname ref`
        query_dir=`dirname query`

        echo $ref_dir >> dirs.txt
        echo $query_dir >> dirs.txt
    >>>
    output {
        Array[String] dirs = read_lines("dirs.txt")
        File ref_dir = "~{dirs[0]}~{ref_name}"
        File query_dir = "~{dirs[1]}~{query_name}"
    }
    runtime {
    	docker: "us-central1-docker.pkg.dev/velina-208320/docker-count/img:latest"
        memory: "100 GB"
        disks: "local-disk 128 SSD"
        cpu: 8
        preemptible: 0
    }
}

task integration {
    input {
        File ref_dir
        File query_dir
        String ref_name
        String query_name
    }
    command <<<
        wget https://raw.githubusercontent.com/mhl34/macosko_scripts/master/cs_integration_seurat.R
   
        Rscript --vanilla cs_integration_seurat.R ~{ref_dir} ~{query_dir} ~{ref_name} ~{query_name}
        # unique identifier
        gcloud storage cp ~{query_dir}/combined_pred.csv gs://macosko_data/leematth/
    >>>
    runtime {
        docker: "us-central1-docker.pkg.dev/velina-208320/docker-count/img:latest"
        memory: "100 GB"
        disks: "local-disk 128 SSD"
        cpu: 8
        preemptible: 0
    }
}