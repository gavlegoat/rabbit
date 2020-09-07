extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=glpk");
    println!("cargo:rebuild-if-changed=wrapper.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // We explicitly whitelist the used functions in order to avoid hundreds of dead code
        // warnings.
        .whitelist_function("glp_create_prob")
        .whitelist_function("glp_set_obj_dir")
        .whitelist_function("glp_add_rows")
        .whitelist_function("glp_add_cols")
        .whitelist_function("glp_load_matrix")
        .whitelist_function("glp_set_row_bnds")
        .whitelist_function("glp_set_col_bnds")
        .whitelist_function("glp_set_obj_coef")
        .whitelist_function("glp_init_smcp")
        .whitelist_function("glp_simplex")
        .whitelist_function("glp_get_status")
        .whitelist_function("glp_get_obj_val")
        .whitelist_function("glp_delete_prob")
        .whitelist_type("glp_smcp")
        .whitelist_var("GLP_MAX")
        .whitelist_var("GLP_UP")
        .whitelist_var("GLP_FR")
        .whitelist_var("GLP_OPT")
        .whitelist_var("GLP_NOFEAS")
        .whitelist_var("GLP_UNBND")
        .whitelist_var("GLP_MSG_OFF")
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
