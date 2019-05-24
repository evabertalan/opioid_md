#execute: vmd -dispdev text -e pbc.tcl -args FOLDER_NAME pbc_log.out 
#eg: FOLDER_NAME = 6b73B

package require pbctools
set code $argv
set files [glob ../../$code/results/namd/step7.*_production.dcd]

mol new ../../$code/results/step5_assembly.xplor_ext.psf type psf
mol off top
set sel [atomselect top all]
set nf [llength $files]

for { set i 0 } { $i < $nf } { incr i } {
    set current_file [lindex $files $i]
    set out_file $current_file-pbc.dcd
    
    animate read dcd $current_file beg 0 end -1 waitfor all
    pbc wrap -all -compound res -center com -centersel protein 

    animate write dcd $out_file
    animate delete all
}
puts 'DONE'
quit 
