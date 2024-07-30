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
            ref = ref_obj
            query = query_obj
            ref_name = ref_name
            query_name = query_name
    }

    call integration {
        input:
            ref = process_anndata.ref_dir
            query = process_anndata.query_dir
            output_name = "~{query_name}_labeled"
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
        python create_seurat_obj.py --ref ~{ref} --query ~{query} --ref_name ~{ref_name} --query_name ~{query_name}
    >>>
    output {
        File ref_dir = "~{ref_name}.qs"
        File query_dir = "~{query_name}.qs"
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
        Rscript --vanilla cs_integration_seurat.R ~{ref_dir} ~{query_dir} ~{ref_name} ~{query_name}
    >>>
}